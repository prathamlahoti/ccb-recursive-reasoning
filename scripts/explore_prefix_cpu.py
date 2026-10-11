"""Bounded CPU investigation; reduced TRM settings, not benchmark evidence."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from ccb.dataset import OfficialEvaluationFirewall, SplitConfig, generate_firewalled_split, instance_fingerprint
from ccb.domains.alien_grid import AlienGridDomain
from ccb.encoding import collate_episodes
from ccb.models.grid_executor import LearnedGridExecutor
from ccb.models.official_trm_ccb import build_ccb_trm_layout
from ccb.training import TrainConfig, build_model, evaluate_batch, seed_everything, train_trm_act_batches


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def run(steps: int, output: Path | None = None) -> dict:
    torch.set_num_threads(1)
    firewall = OfficialEvaluationFirewall.from_official_records('d1')
    domain = AlienGridDomain()
    train = generate_firewalled_split(domain.generate, SplitConfig('cpu_train',(3,),16,91000000), firewall)
    validation = generate_firewalled_split(domain.generate, SplitConfig('cpu_validation',(3,),32,92000000), firewall)
    batch = collate_episodes(train)
    validation_batch = collate_episodes(validation)
    reports = []
    for seed in (7,17):
        for canvas in (3,20):
            for weight in (0.,1.):
                seed_everything(seed)
                config = TrainConfig(model='official_trm_ccb', width=32, layers_or_loops=1,
                    trm_h_cycles=1, trm_l_cycles=1, trm_max_depth=canvas,
                    trm_halt_max_steps=4, trm_halt_exploration_prob=0,
                    learning_rate=.003, weight_decay=0, steps=steps, seed=seed,
                    ema_decay=.99, prefix_frontier_weight=weight, prefix_preservation_weight=weight)
                model = build_model(config, batch.codec)
                started = time.perf_counter()
                _, ema, _, history = train_trm_act_batches(model, [batch]*(steps+1), config)
                reports.append({'seed':seed, 'canvas_depth':canvas, 'prefix_weight':weight,
                    'config':asdict(config),
                    'steps':steps, 'seconds':time.perf_counter()-started,
                    'train_live':evaluate_batch(model,batch), 'train_ema':evaluate_batch(ema.evaluation_model,batch),
                    'validation_ema':evaluate_batch(ema.evaluation_model,validation_batch),
                    'accepted_examples':sum(h['accepted_examples'] for h in history),
                    'first_loss':history[0]['loss'], 'last_loss':history[-1]['loss'],
                    'previous_prefix_states':sum(h.get('previous_prefix_states',0) for h in history),
                    'regressed_prefix_states':sum(h.get('regressed_prefix_states',0) for h in history)})
                print(json.dumps(reports[-1]), flush=True)
                if output is not None:
                    atomic_json(output, {'schema':'ccb_prefix_cpu_exploration_v1',
                        'status':'running', 'trm_runs':reports,
                        'official_evaluation_used':False})
    # Strong, explicitly D1-specific learned baseline. No operation mapping is
    # hard-coded: infer source-cell indices from one-step training targets.
    seed_everything(3)
    transition_domain = AlienGridDomain(randomize_initial_state=True)
    transition_episodes = [transition_domain.generate(depth=1,seed=93000000+i) for i in range(512)]
    transitions = collate_episodes(transition_episodes)
    executor = LearnedGridExecutor()
    optimizer = torch.optim.Adam(executor.parameters(), lr=.1)
    source_indices = (transitions.targets[:,0,:,None] == transitions.initial_state[:,None,:]).long().argmax(-1)
    for _ in range(200):
        optimizer.zero_grad()
        loss = F.cross_entropy(executor.source_logits(transitions.operations[:,0]).reshape(-1,9), source_indices.reshape(-1))
        loss.backward()
        optimizer.step()
    executor_results = {}
    for depth in (5,20,50,100):
        test = collate_episodes([transition_domain.generate(depth=depth,seed=94000000+i) for i in range(100)])
        with torch.no_grad():
            matches = executor(test).logits.argmax(-1) == test.targets
        executor_results[str(depth)] = {'final_exact_accuracy':float(matches[:,-1].all(-1).float().mean()),
                                        'trace_exact_accuracy':float(matches.flatten(1).all(-1).float().mean())}
    coverage = []
    for depth in (5,20,50):
        layout = build_ccb_trm_layout(batch.codec,depth)
        coverage.append({'canvas_depth':depth,'tokens':layout.sequence_length})
    return {'schema':'ccb_prefix_cpu_exploration_v1','status':'complete',
        'purpose':'reduced-model CPU diagnostic; not a benchmark',
        'steps_per_run':steps,'official_evaluation_used':False,'train_examples':16,'train_depth':3,
        'train_instance_hashes':[instance_fingerprint(e) for e in train],
        'validation_instance_hashes':[instance_fingerprint(e) for e in validation],
        'trm_runs':reports,'canvas_sizes':coverage,
        'untrained_output_state_cells_depth21_to50':30*9,
        'd1_learned_executor':{'parameters':executor.routing_logits.numel(),'training_examples':512,'training_steps':200,
            'source_cell_prediction_loss':float(loss.detach()),'tests':executor_results,
            'learned_source_indices':executor.routing_logits.detach().argmax(-1).tolist(),
            'scope':'D1-specific permutation inductive bias; randomized-start generated test instances'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--output',type=Path,default=Path('results/prefix_cpu_exploration_v1/summary.json'))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    result = run(args.steps, args.output)
    atomic_json(args.output, result)
    print('CPU_EXPLORATION_SAVED',args.output,flush=True)
