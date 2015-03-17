__author__ = 'Randall'
from pycompecon import Basis

'''
        just the template from Matlab version.
        Work in progress

'''















## interpolator class
# Defines a class to represent an approximated function
#
# Objects created by this class are a subclass of <basis.m basis>, adding fields to
# identify a function and methods to interpolate, compute Jacobian and Hessian.
#
#
# Apart from the properties inherited from <basis.m basis>, objects of class |interpolator|
# have the following properties:
#
# * |y|: value of interpolated function at basis nodes
# * |c|:  interpolation coefficients
# * |Phi|:   interpolation matrix, evaluated at basis nodes
# * |Phiinv|: inverse of |Phi|
#
# Object of class |funcApprox| have the following methods:
#
# * |funcApprox|: class constructor
# * |updateCoef|: computes interpolation coefficients if |y| is modified
# * |Interpolate|: interpolates the function
# * |Jacobian|: computes the Jacobian of the function
# * |Hessian|: computes the Hessian of the function
#
#
# *To date, only basisChebyshev has been implemented*, so calling the function with |type|
# 'spli' or 'lin' returns an error.
#
# Last updated: November 24, 2014.
#
#
# Copyright (C) 2014 Randall Romero-Aguilar
#
# Licensed under the MIT license, see LICENSE.txt

##
class Interpolator(Basis):
    #properties (Dependent)
    y = None   # value of functions at nodes
    c = None   # interpolation coefficients
    x = None   # basis nodes
    #properties (Access = protected)
    fnodes_ = None # stored values for function at nodes
    coef_ = None # stored coefficients
    fnodes_is_outdated = None# if true, calling "y" updates fnodes_ before returning values
    coef_is_outdated = None# if true, calling "c" updates coef_ before returning values
    #properties (SetAccess = protected)
    Phi = None  # interpolation matrix
    Phiinv = None # inverse of interpolation matrix

    #methods
        
        ## funcApprox

    def __init__(self,fnodes):
        ###
        # Constructor for funcApprox
        #
        #   self = funcApprox(B,fnodes)
        #
        # The inputs are:
        #
        # * |B|: basis object
        # * |fnodes|: matrix, values of the interpolant at basis B nodes
        raise NotImplementedError
        """
        if nargin==0
            return
        end

        # If B is not a 'basis', make one
        if ~isa(B,'basis');  B = basis.make(B);  end

        # Copy basis properties to new object self
        for s = fieldnames(basis)'
            s1 = char(s);
            self.(s1) = B.(s1);
        end

        # Compute interpolation matrix at nodes
        self.Phi = B.Interpolation;

        # Compute inverse of interpolation matrix, (not if Spline)
        if strcmp(self.type, 'Chebyshev')
            self.Phiinv = (self.Phi'*self.Phi)\self.Phi';
        else
            self.Phi = self.Phi(:,:);
        end

        # Default values
        if nargin < 1
            self.y = zeros(size(self.nodes,1),1);
        else
            self.y = fnodes;
        end
    end
    """


    ## SETTERS for "y" and "c"
    #
    #  Since the values of "y" and "c" are mutually dependent, changing one of them requires the other to be
    #  updated by using the interpolation matrix:  y = Phi(x) * c.
    #
    #  To save time, this updating only takes place when the variable is queried.
    @y.setter
    def y(self,value):
        return None
        """
        assert(size(value,1) == size(self.nodes,1), ...
            'funcApprox:y:size',...
            'New value for y must have #d rows (1 per node)',size(self.nodes,1))
        self.fnodes_ = value;
        self.fnodes_is_outdated = false;
        self.coef_is_outdated = true;
    end
    """

    @c.setter
    def c(self,value):
        return None
        """
        assert(size(value,1) == size(self.Phi,2), ...
            'funcApprox:c:size',...
            'New value for c must have #d rows (1 per polynomial)',size(self.Phi,2))
        self.coef_ = value;
        self.coef_is_outdated = false;
        self.fnodes_is_outdated = true;
    end
    """

    ## GETTERS for "y" and "c"
    #
    # If the variable is not outdated, the get methods just returns the stored values. Otherwise, the values are
    # updated using the other variable and then returned.
    @property
    def y(self): #return Fval
        return None
        """
        if self.fnodes_is_outdated
            self.fnodes_ =  reshape(self.Phi * reshape(self.coef_,self.N,prod(self.size)),size(self.coef_));
            self.fnodes_is_outdated = false;
        end
        Fval = self.fnodes_;
    end
    """
    @property
    def c(self): #return Fcoef
        return None
        """
        if self.coef_is_outdated
            switch self.type
                case 'Chebyshev'
                    self.coef_ = reshape(self.Phiinv * reshape(self.fnodes_,self.N,prod(self.size)),[self.M, self.size]);
                case 'Spline'
                    self.coef_ = reshape(self.Phi \ reshape(self.fnodes_,self.N,prod(self.size)),size(self.fnodes_));
            end
            self.coef_is_outdated = false;
        end
        Fcoef = self.coef_;
    end
    """




    ## Size of the object and getter for "x"
    #
    # The size of the object is defined as the number of functions represented by "y" and "c".
    # "x" is just a shortcut for "nodes",

    @property
    def size(self): #return sz
        return None
        """
        if self.fnodes_is_outdated
            sz = size(self.coef_);
        else
            sz = size(self.fnodes_);
        end
        sz(1) = [];
    end
    """

#          function num = numel(self)
#              num = prod(size(self),2);
#          end

    @property
    def x(self): # return xx
        return None
        """
        xx = self.nodes;
    end
    """


    ## Interpolate
    def Interpolate(self,varargin): #return varargout
        return None
        """
        ###
        #   Y = self.Interpolate(x,order,integrate)
        #
        # Interpolates function f: R^d --> R^m at values |x|; optionally computes
        # derivative/integral. It obtains the interpolation matrices by calling
        # |Interpolation| on its basis.
        #
        # Its inputs are
        #
        # * |x|, k.d matrix of evaluation points.
        # * |order|, h.d matrix, for order of derivatives. Defaults to zeros(1,d).
        # * |integrate|, logical, integrate if true, derivative if false. Defaults to
        # false
        #
        # Output |Y| returns the interpolated functions as a k.m.h array, where k is
        # the number of evaluation points, m the number of functions (number of
        # columns in |f.c|), and h is number of order derivatives.

        varargout = cell(1,nargout);

        if nargin <2
            Y = self.y;
            varargout{1} = Y;
            if nargout ==1, return,end
        end

        if nargout > 1 && nargin > 2
            warning('If calling Interpolate with more than one output variables, then ''order'' and ''integrate'' are ignored')
        end

        switch nargout
            case 1
                Phix = self.Interpolation(varargin{:});
                nx = size(Phix,1);  # number of evaluation points
                no = size(Phix,3);  # number of order evaluations
                Y = zeros(nx,self.size,no);

                if issparse(Phix), Phix = full(Phix); end;
                for h = 1:no
                    Y(:,:,h) = Phix(:,:,h) * self.c;
                end


                if self.size==1  # only one function
                    Y = squeeze(Y);
                end

                varargout{1} = Y;
            case 2
                if nargin<2;
                    x = self.nodes;
                else
                    x = varargin{1};
                end
                [varargout{2}, varargout{1}] = Jacobian(self, x,[],[],false);
            case 3
                if nargin<2;
                    x = self.nodes;
                else
                    x = varargin{1};
                end
                [varargout{2}, varargout{1}] = Jacobian(self, x,[],[],false);
                varargout{3} = Hessian(self,x);
        end


    end #Evaluate
    """


    ## Jacobian
    def Jacobian(self, x, indx,indy, permuted): #return DY, Y
        return None
        """
        ###
        #   [DY,Y] = self.Jacobian(x, index)
        #
        # Computes the Jacobian of the approximated function f: R^d --> R^m.
        #
        # Inputs:
        #
        # * |x|, k.d matrix of evaluation points.
        # * |index|, 1.d boolean, take partial derivative only wrt variables with
        # index=true. Defaults to true(1,d).
        #
        # Outputs
        #
        # * |DY|, k.m.d1 Jacobian matrix evaluated at |x|, where d1 = sum(index)
        # * |Y|, k.m interpolated function (same as Y = B.Evaluate(c,x), provided
        # for speed if both Jacobian and
        # funcion value are required).

        ###
        # Restrict function to compute
        if nargin<5 || isempty(indy)
            COEF = reshape(self.c,self.M,[]);
        elseif isa(indy,'cell')
            COEF = self.c(:,indy{:});
            COEF = reshape(COEF,self.M,[]);
        else
            COEF = self.c(:,indy);
        end



        ###
        # Solve for the one-dimensional basis

        if self.d==1
            if nargout == 1
                Phix  = self.Interpolation(x,1,false);
                DY = Phix * COEF;
            else
                Phix = self.Interpolation(x,[1;0],false);
                DY = Phix(:,:,1) * COEF;
                Y = Phix(:,:,2) * COEF;
            end

            return
        end

        ###
        # Solve for the multi-dimensional basis

        # Check validity of input x
        assert(size(x,2) == self.d, 'In Jacobian, class basis: x must have d columns')

        ###
        # Keep track of required derivatives: Required is logical with true indicating derivative is required
        if nargin<3 || isempty(indx)
            Required = true(1,self.d);
            indx = 1:self.d;
        elseif numel(indx) < self.d # assume that index have scalars of the desired derivatives
            Required = false(1,self.d);
            Required(indx) = true;
        else # assume that index is nonzero for desired derivatives
            Required = logical(indx);
            indx = find(indx);
        end





        if nargin<5
            permuted = true;
        end




        nRequired = sum(Required);

        ###
        # HANDLE NODES DIMENSION
        Nrows = size(x,1);

        ###
        # HANDLE POLYNOMIALS DIMENSION
        C = self.opts.validPhi;
        Ncols = size(C,1);

        ###
        # Compute interpolation matrices for each dimension

        Phi0 = zeros(Nrows,Ncols,self.d);
        Phi1 = zeros(Nrows,Ncols,self.d);

        for k = 1:self.d
            if Required(k)
                PhiOneDim = self.B1(k).Interpolation(x(:,k),...
                    [0 1],...
                    false);
                Phi01 = PhiOneDim(:,C(:,k),:);
                Phi0(:,:,k) = Phi01(:,:,1); # function value
                Phi1(:,:,k) = Phi01(:,:,2); # its derivative
            else
                PhiOneDim = self.B1(k).Interpolation(x(:,k),...
                    0,...
                    false);
                Phi0(:,:,k) = PhiOneDim(:,C(:,k));
            end
        end

        ###
        # Compute the Jacobian
        # Preallocate memory
        DY  = zeros(Nrows,prod(self.size),nRequired);

        # Multiply the 1-dimensional bases

        for k=1:nRequired
            Phik = Phi0;
            Phik(:,:,indx(k)) = Phi1(:,:,indx(k));
            Phix = prod(Phik,3);
            DY(:,:,k) = Phix * COEF;
        end

        ###
        # Compute the function if requested
        if nargout > 1
            Y = prod(Phi0,3) * COEF;
        end


        ###
        # Permute the Jacobian, if requested
        #
        # Permute the resulting Jacobian so that indices reflect "usual" 2D Jacobian, with 3rd dimension used for
        # different observations.
        #
        # * i = function
        # * j = variable wrt which function is differentiated
        # * k = observation

        if permuted
            Y = permute(Y, [2,3,1]);
            DY = permute(DY, [2 3 1]);
        end


    end # Jacobian
    """


    ## Hessian
    def Hessian(self,x,indy): # return HY
        return None
        """
        ###
        #   Hy = self.Hessian(x,c)
        #
        # Computes the Hessian of a function approximated by basis |B| and coefficients |c|.
        #
        # Its input is
        #
        # * |x|, k.d matrix of evaluation points.
        #
        # Its output |Hy| returns the k.m.d.d Hessian evaluated at |x|.


        if nargin<3 || isempty(indy)
            COEF = reshape(self.c,self.N,[]);
        else
            COEF = reshape(self.c(:,indy),self.n,[]);
        end





        order = repmat({[0 1 2]'},1,self.d);
        order = gridmake(order{:});
        order = order(sum(order,2)==2,:);

        Phix = self.Interpolation(x,order,false);


        nx = size(x,1);     # number of evaluated points

        #Dy = squeeze(Dy);

        Hy = zeros(nx,prod(self.size),self.d,self.d);

        for k = 1:size(order,1)
            i = find(order(k,:));
            if numel(i)==1
                Hy(:,:,i,i) = Phix(:,:,k) * COEF;#  Dy(:,k);
            else
                Hy(:,:,i(1),i(2)) = Phix(:,:,k) * COEF; #Dy(:,k);
                Hy(:,:,i(2),i(1)) = Hy(:,:,i(1),i(2)); #Dy(:,k);
            end
        end
    end # Hessian
    """

    ## subsref


    """
    function varargout = subsref(F,s)
        NARGOUT = max(1,nargout);

        switch s(1).type
            case '.'
                [varargout{1:NARGOUT}] = builtin('subsref',F,s);
                return
            case '()'
                [varargout{1:NARGOUT}]  = F.Interpolate(s(1).subs{:}); # TODO: Vargout
                return
            case '{}'
                if numel(s)==1
                    sref = copy(F);
                    if F.fnodes_is_outdated;
                        sref.c = F.coef_(:,s.subs{:});
                    else
                        sref.y = F.fnodes_(:,s.subs{:});
                    end
                    varargout{1} = sref;
                    return
                else
                    switch s(2).type
                        case '()'
#                                 warning('Work-in-progress: should call interpolate with restricted values of coefficients, instead of making copy of F');
                            F2 = subsref(F,substruct('{}',s(1).subs));
                            [varargout{1:NARGOUT}]  = F2.Interpolate(s(2).subs{:}); # TODO: Vargout
                            return



                        case '.'
                            switch s(2).subs
                                case {'y','c'}
                                    if numel(s)==2
                                        s2 = substruct('.',s(2).subs,'()',[{':'} s(1).subs]);
                                        varargout{1} = builtin('subsref',F,s2);
                                        return
                                    else
                                        error('Too many index levels');
                                    end
                                otherwise
                                    varargout{1} = builtin('subsref',F,s(2));
                            end
                    end
                end
        end
    end

    ## subsasgn
    function F = subsasgn(F,s,val)

        switch s(1).type
            case '.'
                F = builtin('subsasgn',F,s,val);
                return
            case '()'
                if numel(s)==1
                    F.y(:,s.subs{:}) = val;
#                         s2 = substruct('.','y','()',[{':'},s.subs]);
#                         F = builtin('subsasgn',F,s2,val);
                    return
                else
                    switch s(2).type
                        case '.'

                            switch s(2).subs
                                case {'y','c'}
                                    if numel(s)==2
                                        s2 = substruct('.',s(2).subs,'()',[{':'} s(1).subs]);
                                        F = builtin('subsasgn',F,s2,val);
                                        return
                                    else
                                        error('Too many index levels');
                                    end
                                otherwise
                                    F = builtin('subsasgn',F,s(2),val);
                            end
                    end
                end
            case '{}'
                if numel(s)==1
                    s2 = substruct('.','c','()',[{':'},s.subs]);
                    F = builtin('subsasgn',F,s2,val);
                    return
                else
                    error('On assigning coefficients with {i,j} indexing, no further indexing allowed')
                end

        end
    end
    end
    
end

"""