"""Independent frozen-role execution controller: never delegates to native .train()."""
from stage20b_runtime_contract import augmentation_seed, validate_schedule


class Stage20BDetectorController:
    """Shared future loop skeleton; callbacks isolate GPU execution from R0 simulation.

    No validator, stopper, best checkpoint, OOM retry, schedule resampling or
    implicit accumulation. Actual optimizer calls belong only to an explicitly
    authorized future attempt callback; R0 supplies metadata callbacks only.
    """
    def __init__(self, cfg, order):
        validate_schedule(cfg, order)
        self.cfg, self.order = cfg, order

    def run(self, draw, attempt, epoch_end=None, cleanup=None):
        counts = {'epochs': 0, 'draws': 0, 'scheduled_optimizer_attempts': 0}
        try:
            for epoch in self.order['epochs']:
                start = 0
                for end, length in zip(epoch['optimizer_attempt_after_positions_0based'], epoch['accumulation_group_lengths']):
                    if end - start + 1 != length:
                        raise RuntimeError('DETECTOR_RUNTIME_SCHEDULE_INVALID')
                    for pos in range(start, end + 1):
                        draw(epoch['epoch_0based'], pos, epoch['role_sequence'][pos],
                             epoch['augmentation_seeds'][pos], length)
                        counts['draws'] += 1
                    attempt(epoch['epoch_0based'], end, length)
                    counts['scheduled_optimizer_attempts'] += 1
                    start = end + 1
                counts['epochs'] += 1
                if epoch_end is not None:
                    epoch_end(epoch['epoch_0based'])
            return counts
        finally:
            # Same no-validation exit for normal completion AND exceptions.
            if cleanup is not None:
                cleanup()

    def learning_rates(self, epoch, position, param_groups):
        """Native pinned linear LR/bias warmup; never changes accumulation64/tail24."""
        args = self.cfg['detector']
        factor = max(1 - epoch / args['epochs'], 0) * (1 - args['lrf']) + args['lrf']
        ni, nw = epoch * 216 + position, round(args['warmup_epochs'] * 216)
        values = []
        for group in param_groups:
            target = group.get('initial_lr', args['lr0']) * factor
            initial = args['warmup_bias_lr'] if group.get('param_group') == 'bias' else 0.0
            values.append(initial + (target - initial) * ni / nw if nw > 0 and ni < nw else target)
        return values

    @staticmethod
    def backward(loss, group_length, scaler=None):
        # Applies to tail24 too: mean over the actual group's draws, not /64.
        normalized = loss.sum() / group_length
        (scaler.scale(normalized) if scaler is not None else normalized).backward()


def accumulation_dryrun(cfg, order):
    controller = Stage20BDetectorController(cfg, order)
    rows, gradients = [], []
    state = {'gradient': 0.0, 'draws_in_group': 0}

    def draw(epoch, pos, role, seed, length):
        # d/dw (w * (position+1)) = position+1; analytic no-optimizer mean check.
        state['gradient'] += (pos + 1) / length
        state['draws_in_group'] += 1
        assert seed == augmentation_seed(cfg['protocol'], epoch, pos)

    def attempt(epoch, pos, length):
        expected = sum(range(pos - length + 2, pos + 2)) / length
        assert state['draws_in_group'] == length and abs(state['gradient'] - expected) < 1e-9
        rows.append({'epoch': epoch, 'position': pos, 'draws': length, 'normalizer': length})
        gradients.append(state['gradient'])
        state.update(gradient=0.0, draws_in_group=0)

    counts = controller.run(draw, attempt)
    return {'status': 'PASS', **counts, 'attempt_groups': rows,
            'analytic_normalization_pass': True, 'first_epoch_mean_gradients': gradients[:4],
            'actual_optimizer_steps': 0, 'simulation_only': True,
            'actual_successful_amp_updates': 'NOT_RUN'}
