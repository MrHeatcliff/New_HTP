import collections
import hashlib
import json
from pathlib import Path
from functools import partial as bind

import elements
import embodied
import numpy as np

from .paper_artifacts import PaperArtifactWriter


def _checkpoint_sha256(directory):
  """Hash a completed checkpoint directory including names and contents."""
  directory = Path(str(directory))
  digest = hashlib.sha256()
  for path in sorted(x for x in directory.rglob('*') if x.is_file()):
    digest.update(str(path.relative_to(directory)).encode())
    with path.open('rb') as stream:
      for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
  return digest.hexdigest()


class TraceableRatio:

  def __init__(self, ratio):
    self.ratio = ratio
    self.prev = None

  def __call__(self, step):
    step = int(step)
    before = self.prev
    if self.ratio == 0:
      repeats = 0
    elif self.ratio < 0:
      repeats = 1
    elif self.prev is None:
      self.prev = step
      repeats = 1
    else:
      repeats = int((step - self.prev) * self.ratio)
      self.prev += repeats / self.ratio
    return repeats, before, self.prev


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  exact_action_budget = bool(getattr(args, 'exact_env_action_budget', False))
  env_action_steps = elements.Counter()
  reset_callbacks = elements.Counter()
  replay_insertions = elements.Counter()
  raw_ale_frames = elements.Counter()
  if exact_action_budget:
    if int(args.envs) != 1:
      raise RuntimeError(
          'Exact action milestones require run.envs=1; vector stepping can '
          'jump over a milestone and must not be labeled exact.')
    action_milestones = tuple(map(int, args.action_milestones))
    if tuple(sorted(set(action_milestones))) != action_milestones:
      raise ValueError(f'Invalid action milestones: {action_milestones}')
    if int(args.steps) != action_milestones[-1]:
      raise ValueError((args.steps, action_milestones))
  else:
    action_milestones = ()
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()
  paper = PaperArtifactWriter(logdir, args)
  paper.bind_runtime_counters(
      env_action_steps, reset_callbacks, replay_insertions, raw_ale_frames,
      optimizer_updates=lambda: int(agent.n_updates))

  batch_steps = args.batch_size * args.batch_length
  should_train = TraceableRatio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3 and args.log_policy_video:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      score = result.pop('score')
      length = result.pop('length')
      logger.add({'score': score, 'length': length}, prefix='episode')
      paper.write_episode(step, score, length)
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  def actioncountfn(tran, worker):
    executed = bool(tran['log/action_executed'])
    if executed:
      env_action_steps.increment()
    else:
      reset_callbacks.increment()
    raw_ale_frames.increment(int(tran.get('log/raw_ale_frames', 0)))
  driver.on_step(actioncountfn)
  driver.on_step(lambda tran, _: policy_fps.step())
  def replayfn(tran, worker):
    replay.add(tran, worker)
    replay_insertions.increment()
  driver.on_step(replayfn)
  driver.on_step(logfn)
  driver.on_step(lambda tran, worker: paper.write_action_trace(
      step, tran, worker, int(getattr(agent, 'n_updates', 0))))

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)
  updates_before_event = [0 for _ in range(args.envs)]

  def trainfn(tran, worker):
    updates_before_event[worker] = int(getattr(agent, 'n_updates', 0))
    if len(replay) < args.batch_size * args.batch_length:
      paper.write_update_event(
          step, requested_updates=0, executed_updates=0,
          optimizer_updates_cumulative=int(getattr(agent, 'n_updates', 0)),
          is_prefill=True, is_compile_only=False,
          scheduler_accumulator_before=should_train.prev,
          scheduler_accumulator_after=should_train.prev)
      return
    requested, sched_before, sched_after = should_train(step)
    executed = 0
    for _ in range(requested):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      before = int(getattr(agent, 'n_updates', 0))
      paper.write_batch_trace(step, executed, batch, before)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      executed += 1
      paper.write_batch_trace(
          step, executed - 1, batch, before,
          optimizer_updates_after=int(getattr(agent, 'n_updates', 0)),
          metrics={f'train/{k}': v for k, v in mets.items()})
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
    paper.write_update_event(
        step, requested_updates=requested, executed_updates=executed,
        optimizer_updates_cumulative=int(getattr(agent, 'n_updates', 0)),
        is_prefill=False, is_compile_only=False,
        scheduler_accumulator_before=sched_before,
        scheduler_accumulator_after=sched_after)
  driver.on_step(trainfn)

  def actiontracefn(tran, worker):
    paper.write_env_action_event(
        step, tran, worker, env_action_steps, reset_callbacks,
        replay_insertions,
        updates_before_event[worker], int(getattr(agent, 'n_updates', 0)))
  driver.on_step(actiontracefn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if exact_action_budget:
    cp.env_action_steps = env_action_steps
    cp.reset_callbacks = reset_callbacks
    cp.driver_callbacks = step
    cp.replay_insertions = replay_insertions
    cp.raw_ale_frames = raw_ale_frames
    cp.action_metadata = elements.Saveable(
        save=lambda: {
            'env_action_steps': int(env_action_steps),
            'driver_callbacks': int(step),
            'reset_callbacks': int(reset_callbacks),
            'replay_insertions': int(replay_insertions),
            'optimizer_updates': int(agent.n_updates),
            'raw_ale_frames': int(raw_ale_frames),
            'milestone': int(env_action_steps),
        },
        load=lambda data: None)
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  completed_milestones = {
      milestone for milestone in action_milestones
      if milestone <= int(env_action_steps)}
  exact_manifest_path = logdir / 'paper_artifacts/action_checkpoints_manifest.json'

  def milestonefn(tran, worker):
    if not exact_action_budget or not bool(tran['log/action_executed']):
      return
    current = int(env_action_steps)
    if current not in action_milestones or current in completed_milestones:
      return
    folder = f'env_action_steps_{current:09d}'
    target = logdir / 'ckpt' / folder
    cp.save(target)
    checkpoint_hash = _checkpoint_sha256(target)
    (logdir / 'ckpt/latest').write_text(folder)
    completed_milestones.add(current)
    rows = []
    if exact_manifest_path.exists():
      rows = json.loads(exact_manifest_path.read_text())
    rows.append({
        'env_action_steps': current,
        'driver_callbacks': int(step),
        'reset_callbacks': int(reset_callbacks),
        'replay_insertions': int(replay_insertions),
        'optimizer_updates': int(agent.n_updates),
        'reset_callbacks_per_env_action': (
            int(reset_callbacks) / current if current else None),
        'optimizer_updates_per_env_action': (
            int(agent.n_updates) / current if current else None),
        'raw_ale_frames': int(raw_ale_frames),
        'checkpoint': str(target),
        'checkpoint_hash': checkpoint_hash,
        'milestone': current,
        'checkpoint_action_milestone': current,
        'method': paper.method,
        'git_commit': paper.code_commit,
        'resolved_config_hash': paper.config_hash,
        **paper.htp_metadata(),
    })
    exact_manifest_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + '\n')

  driver.on_step(milestonefn)

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  budget_counter = env_action_steps if exact_action_budget else step
  while budget_counter < args.steps:

    driver(policy, steps=1 if exact_action_budget else 10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      train_stats = train_agg.result()
      ep_stats = epstats.result()
      replay_stats = replay.stats()
      usage_stats = usage.stats()
      fps_stats = {
          'fps/policy': policy_fps.result(),
          'fps/train': train_fps.result(),
      }
      timer_stats = {'timer': elements.timer.stats()['summary']}
      logger.add(train_stats)
      logger.add(ep_stats, prefix='epstats')
      logger.add(replay_stats, prefix='replay')
      logger.add(usage_stats, prefix='usage')
      logger.add(fps_stats)
      logger.add(timer_stats)
      paper_stats = {}
      paper_stats.update(train_stats)
      paper_stats.update({f'epstats/{k}': v for k, v in ep_stats.items()})
      paper_stats.update({f'replay/{k}': v for k, v in replay_stats.items()})
      paper_stats.update({f'usage/{k}': v for k, v in usage_stats.items()})
      paper_stats.update(fps_stats)
      paper_stats.update(timer_stats)
      paper.write_train_metrics(step, paper_stats)
      logger.write()

    if not exact_action_budget and should_save(step):
      cp.save()

  if exact_action_budget:
    assert int(env_action_steps) == int(args.steps)
    assert int(args.steps) in completed_milestones
  else:
    cp.save()
  paper.finalize(
      step, checkpoint_path=(
          logdir / 'ckpt' / f'env_action_steps_{int(env_action_steps):09d}'
          if exact_action_budget else logdir / 'ckpt'))
  logger.close()
