from pycompecon import BasisChebyshev
import numpy as np
__author__ = 'Randall'



class Basis:
    """
      A multivariate interpolation basis
    """

    def __init__(self, n, a, b, **options):
        n = np.array([n] if np.isscalar(n) else n, 'int')
        a = np.array([a] if np.isscalar(a) else a)
        b = np.array([b] if np.isscalar(b) else b)


        d = a.size  # dimension of basis

        # todo use same number of nodes in all dimension if n is scalar
        if not np.all(a < b):
            raise ValueError('All lower bounds must be less than their corresponding upper bound')

        # Make default options dictionary
        opts = dict(
            type='chebyshev',
            nodetype='gaussian',
            k=3,
            method='tensor',
            qnode=None,
            qdegree=None,
            varnames=["V{}".format(dim) for dim in range(d)]
        )

        valid_opts = dict(
            type={'chebyshev', 'spline'},
            nodetype={'gaussian', 'lobatto', 'endpoint','cardinal'},   #todo: cardinal is only valid in spline basis
            k=range(min(n)),
            method={'tensor', 'smolyak', 'complete', 'cluster', 'zcluster'},
            qnode=range(min(n)),
            qdegree=range(min(n)),
        )

        # get variable names, if provided
        if 'varnames' in options.keys():
            if len(options['varnames']) == d:
                opts["varnames"] = options['varnames']
            else:
                print('If provided, option varnames must have {} elements'.format(d))
            del options['varnames']

        # Read user options
        for opt, val in options.items():
            if opt not in opts.keys():
                print('Unknown option {} with value {}; ignoring'.format(opt, val))  # todo: make this a warning
            elif val not in valid_opts[opt]:
                print('Value {} is not a valid option for {}; ignoring'.format(val,opt))  # todo: make this a warning
            else:
                opts[opt] = val

        # Validate options for chebyshev basis of several dimensions
        if d > 1 and opts['type'] == 'chebyshev':
            if opts['method'] in ['complete', 'cluster', 'zcluster']:
                if opts['qnode'] is None:
                    opts['qnode'] = 0
                if opts['qdegree'] is None:
                    opts['qdegree'] = max(n) - 1
            elif opts['method'] == 'smolyak':
                n_valid = 2 ** np.ceil(np.log2(n - 1)) + 1
                if np.any(n != n_valid):
                    # todo: make this a warning
                    print('Warning: For smolyak expansion, number of nodes should be n = 2^k+1 for some k=1,2,...')
                    print('Adjusting nodes\n {:7s}  {:7s}'.format('old n', 'new n'))
                    for n1, n2 in zip(n,n_valid):
                        print('{:7.0f} {:7.0f}'.format(n1, n2))
                    n = np.array(n_valid,'int')
                if opts['nodetype'] != 'lobatto':
                    opts['nodetype'] = 'lobatto'  # todo issue warning
                if opts['qnode'] is None:
                    opts['qnode'] = 2
                if opts['qdegree'] is None:
                    opts['qdegree'] = opts['qnode']

        # make list of 1-basis
        B1 = []
        nodetype, varnames = opts['nodetype'], opts['varnames']

        if opts['type'] == 'chebyshev':
            for i in range(d):
                B1.append(BasisChebyshev(n[i], a[i], b[i], nodetype, varnames[i]))

        # Pack value in object
        self.a = a
        self.b = b
        self.n = n
        self.d = d
        self.opts = opts
        self._B1 = B1
        self.type = opts['type']

        # todo: expand basis

    def expandBasis(self):
        raise Exception('NOT YET IMPLEMENTED')

    def interpolation(self,x,order):
        raise Exception('NOT YET IMPLEMENTED')

    def plot(self):
        raise Exception('NOT YET IMPLEMENTED')

    def __repr__(self):
        raise Exception('NOT YET IMPLEMENTED')


def SmolyakGrid(n,qnode,qdegree):
    raise Exception('NOT YET IMPLEMENTED')