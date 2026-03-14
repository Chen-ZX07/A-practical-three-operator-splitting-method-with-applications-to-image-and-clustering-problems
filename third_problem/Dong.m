function  out = Dong(m,n,Omega,b,para,stop)
flag = exist('para');
if(flag && isfield(para,'detail'))
    detail = para.detail;
else
    detail = 0;
end

u = zeros(m,n);
X = zeros(m,n);   Y = X;    Z = X; 
a1 =-X;          a2 =-Y;    a3 =-Z;
M = zeros(m,n);   M(Omega) = b;      % Given matrix M
alphabar = 1.2*(3/(4*1.8))+10^(-9)/(4*1.8);
% alphabar = sqrt(n)*1.001;
% alphabar = 1.9;
alpha1 = 1.8;
alpha2 = 1.8;   %1.2
alpha3 = 1.8;
theta = 0.8;
% alphabar= 3;
% alpha1=1;
% theta=1;   %1 - 1.8    0.1jiange   

% alpha2 = alpha;
% alphabar = 3/alpha2;
% % theta = 1;


% theta = 1;

sv = 100;
lh = 1;
mu = para.mu;       tau = para.tau;   % Regularized parameters
sct = para.sct;

1

for iter = 1:stop.Max
    L=M-(X+Y+Z);
    ubar = u - L/alphabar;
    Xbar = X - (a1 + ubar)/alpha1;
    Ybar = Y - (a2 + ubar)/alpha2;
    Zbar = Z - (a3 + ubar)/alpha3;
    
    X1=X-Xbar; Y1=Y-Ybar; Z1=Z-Zbar; u1=u-ubar; 
    L = M-(Xbar+Ybar+Zbar);
    nx=norm(X1,'fro')^2;
    ny=norm(Y1,'fro')^2;
    nz=norm(Z1,'fro')^2;
    %g1 = alphabar*(nx+ny+nz)+sum(sum(L.*u1))
    g1 = alpha1*nx+alpha2*ny+alpha3*nz+trace(L'*u1);
    g2 = nx+ny+nz+norm(L,'fro')^2;
    gamma = theta*g1/g2;

    
    T_X = X+(a1-gamma*X1)/alpha1;
    if choosvd(n,sv) == 1                 % Employ the PROPACK SVD
        [U,D,V] = lansvd( T_X, sv, 'L');   % fprintf('lansvd   ');
        lh = lh + 1;
    else
        [U,D,V] = svd( T_X, 'econ' );      % fprintf('fullsvd  ');
    end
    nx=sct/alphabar;
    D = diag(D);    index = find( D > nx );  % shrinkage operator
    D = diag( D(index) - nx );
    Xk = U(:,index) * D * V(:,index)';
    out.Rankr(iter) = length( index );       % Record the rank
    svp = out.Rankr(iter);
    if svp < sv
        sv = min( svp + 1, n);
    else
        sv = min( svp + round(0.04*n), n);
    end 
    
    T_Y = Y+(a2-gamma*Y1)/alpha2;
    Yk = sign(T_Y).*max(abs(T_Y)-tau*nx,0);
    out.SP(iter) = sum( abs(Yk(:)) > 1e-2 );
    
    
    T_Z = Z+(a3-gamma*Z1)/alpha3;
    if strcmp(para.model,'unconstrained')
        factor = mu / ( nx + mu );
    else
        znorm = norm( T_Z(Omega), 'fro' );
        factor = min( znorm, para.delta)/ znorm;
    end
    T_Z(Omega) = factor.*(T_Z(Omega));     
    Zk = T_Z;
    
    u = u - gamma*L;
    
    a1 = alphabar*(X-Xk)+a1-gamma*X1;
    a2 = alphabar*(Y-Yk)+a2-gamma*Y1;
    a3 = alphabar*(Z-Zk)+a3-gamma*Z1;
    
    out.error = max( norm(Xk - X,'fro')/(1+norm(Xk,'fro')), ...
        norm(Yk - Y,'fro')/(1+norm(Yk,'fro')) + norm(Zk - Z,'fro')/(1+norm(Zk,'fro')));
    out.LowRank = Xk;
    out.Sparse = Yk;
    out.M = M;
    out.iter = iter;
    X=Xk;
    Y=Yk;
    Z=Zk;
    
    [~,VV,~] = svd(X, 'econ');
    out.obj(iter) = sum(diag(VV)) + tau*sum(abs(Y(:))) ...
        + 0.5*norm(Z(Omega),'fro')^2/mu;
    if strcmp(stop.rule,'TOL')
        if out.error <= stop.eps || iter >= stop.Max
            out.LowRank = X;   out.Sparse = Y;    out.M = M;
            out.iter = iter;
            return;
        end
    end
    
    if (detail)    %% Display the details of the procedure
        fprintf('It: %3d cpu: %5.2f  rank: %3d spa: %7d stopic: %4.2e \n',...
            iter,out.Rankr(iter),out.SP(iter),out.error);
    end
    
end

