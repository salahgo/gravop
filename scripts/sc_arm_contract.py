"""P13 part 1 -- the deployment contract, per arm and per task.

The paper says arm C "needs no t_sing". True, and incomplete: what an arm needs
at deployment depends on what you are asking it to do, and the two tasks people
actually have make different demands.

  TASK 1  FORWARD PREDICTION. "Given initial data and parameters, what does the
          field look like at physical time t?" Nothing about the system at time
          t is available -- that is the question.
  TASK 2  STATE RECONSTRUCTION. "Given a system observed now, with a measured
          value of the observable, what is the rest of its field?" This is the
          gravothermal use case: one halo, one epoch, a measured central
          density.

Arm C is excellent at task 2 and cannot do task 1 unaided: its coordinate is a
contrast that must be MEASURED, and at a physical time nobody has measured
anything. That is not a caveat, it is the scope of the method, and it is why
clock recovery (P14) is the piece that would extend it.

Every arm's operator takes the initial field as input, so that requirement is
shared and is not what separates them. What separates them is the conditioning
scalar: where it comes from, and what it costs to get.

Output: results/statecond/arm_contract.json
"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'results', 'statecond')

ARMS = {
    'A_tau_oracle': {
        'conditioning': 'tau = t / t_sing, true t_sing supplied',
        'needs_beyond_initial_field': ['t_sing, exactly'],
        'source': 'none: t_sing is defined by the singularity being predicted',
        'acquisition_cost': 'not obtainable',
        'task1_forward_prediction': 'NOT DEPLOYABLE',
        'task2_state_reconstruction': 'NOT DEPLOYABLE',
        'role': 'upper bound and baseline; the number the literature reports',
    },
    'B_tau_deployed': {
        'conditioning': 'tau = t / t_sing_hat, t_sing_hat from a learned head',
        'needs_beyond_initial_field': [],
        'source': 'a scalar head on the same initial field the operator takes',
        'acquisition_cost': 'one extra forward pass',
        'task1_forward_prediction': 'DEPLOYABLE',
        'task2_state_reconstruction': ('DEPLOYABLE only if the observation '
                                       'epoch is known in physical time'),
        'known_error': 't_sing predicted to 1.8-7.1% (median, per system)',
        'role': 'the standard recipe; P15 shows it is not the best time arm',
    },
    'C_state': {
        'conditioning': 'phi = log10(A(t)/A(0)) and the branch flag b',
        'needs_beyond_initial_field': [
            'A(t), the observable measured at the query time',
            'A(0), the observable at the INITIAL epoch',
            'b, whether A is past its minimum'],
        'source': ('A(t) from the observation; A(0) from an earlier epoch or a '
                   'prior; b from history, or inferred from one field (P13)'),
        'acquisition_cost': ('one measurement at the query time, one at t=0, '
                             'and a branch determination'),
        'task1_forward_prediction': ('NOT DEPLOYABLE unaided: phi at a given '
                                     'physical time is unknown without a clock '
                                     '(this is what P14 would supply)'),
        'task2_state_reconstruction': 'DEPLOYABLE, subject to the contract below',
        'contract': ('A(0) to within the per-system tolerance measured in '
                     'contract.json; the observable resolved at the aperture '
                     'and resolution used there; b inferred at the measured '
                     'rate'),
        'role': 'the method of this paper',
    },
    'D_time_direct': {
        'conditioning': 'standardised log10 t',
        'needs_beyond_initial_field': [],
        'source': 'the query itself: the user knows what time they are asking about',
        'acquisition_cost': 'none',
        'task1_forward_prediction': 'DEPLOYABLE',
        'task2_state_reconstruction': ('DEPLOYABLE only if the observation '
                                       'epoch is known in physical time'),
        'role': ('the do-nothing-clever baseline; P15 shows it beats arm B in '
                 'all five systems and beats arm C in burgers'),
    },
}

NOTES = {
    'shared': ('every arm\'s operator consumes the initial field and the '
               'physical parameters; that requirement is common and is not '
               'what distinguishes the arms'),
    'why_two_tasks': ('arm C answers "what does the system look like at '
                      'contrast phi", not "at time t". The two coincide only '
                      'through the map t <-> phi, which is exactly what arm C '
                      'declines to model'),
    'measured_in': 'results/statecond/contract.json (P13), direct.json (P15)',
}


def main():
    os.makedirs(OUT, exist_ok=True)
    doc = {'arms': ARMS, 'notes': NOTES}
    p = os.path.join(OUT, 'arm_contract.json')
    json.dump(doc, open(p, 'w'), indent=1)

    w = max(len(k) for k in ARMS)
    print(f"{'arm':{w}s} | task 1 forward      | task 2 reconstruction")
    print('-' * (w + 46))
    for k, v in ARMS.items():
        t1 = v['task1_forward_prediction'].split(':')[0][:19]
        t2 = v['task2_state_reconstruction'].split(':')[0][:24]
        print(f'{k:{w}s} | {t1:19s} | {t2}')
    print('\nwrote', p)


if __name__ == '__main__':
    main()
