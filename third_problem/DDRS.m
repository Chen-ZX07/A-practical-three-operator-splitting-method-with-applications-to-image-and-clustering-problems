% Refer to: A distributed Douglas-Rachford Splitting Method for
% Multiple-Block Separable Convex Programs
% Authors: Hongjin He, Deren Han, Hongkun Xu, Xiaoming Yuan
% Email to: hehj2003@163.com
% Code for Robust Principal Component Analysis

function  out = DDRS(m,n,Omega,b,para,stop)

flag = exist('para');
if(flag && isfield(para,'detail'))
    detail = para.detail;
else
    detail = 0;
end

mu = para.mu;       tau = para.tau;   % Regularized parameters
beta = para.beta;                     % Penalty parameter
gamma = para.gamma;                   % Relaxation parameter
sct = para.sct;                  % Scaling parameter of objective function

X = zeros(m,n);   Y = X;    Z = X;    Lam = X;   % Initial points
SX = X;          SY = Y;   SZ = Z;    % Subgradients w.r.t. X, Y, Z;

M = zeros(m,n);   M(Omega) = b;      % Given matrix M

if strcmp(stop.rule,'FIXED')
    out.Rankr = zeros(stop.Max,1);   % Record the rank history of matrix
    out.Time  = zeros(stop.Max,1);   % Record parallel CPU time
    out.TimeT = zeros(stop.Max,1);   % Record sequetial time
    out.obj   = zeros(stop.Max,1);   % Record the objective function value
    out.SP    = zeros(stop.Max,1);   % Record the sum of absolute value of matrix
end
3
sv = 100;  lh = 1;    time = 0;  timeT = 0 ;
for iter = 1 : stop.Max
    tic;  % time0 = cputime;
    et = beta.*( X + Y + Z - M);   % Compute \tilde e(\x, beta);
    BarLam = Lam - et;             % Compute \bar \lambda;
    ex = beta.*( SX - BarLam );    % Compute e_1(X^k,\bar\lambda^k);
    ey = beta.*( SY - BarLam );    % Compute e_2(Y^k,\bar\lambda^k);
    ez = beta.*( SZ - BarLam );    % Compute e_3(Z^k,\bar\lambda^k);
    
    EN = ex(:)'*ex(:) + ey(:)'*ey(:) + ez(:)'*ez(:);  % ||e_i(x_i)||^2
    EXYZ = ex + ey + ez;           % Compute e_1 + e_2 + e_3;
    phi = 0.5*EN + et(:)'*et(:) - beta.*( et(:)'*EXYZ(:));
    etxyz = et - beta.*EXYZ;       % Compute ( et - e1 - e2 - e3 );
    psi = EN + etxyz(:)'*etxyz(:);
    alpha = gamma * phi/psi;            % Compute the step size;
    %     fprintf(' alpha: %2.3f ', alpha);
    time0 = toc; %time0 = cputime - time0;
    
    %%========= Compute the x-subproblem ============================
    tic;  % timex = cputime;
    wx = X + beta.*SX - alpha.*ex;
    if choosvd(n,sv) == 1                 % Employ the PROPACK SVD
        [U,D,V] = lansvd(wx, sv, 'L');   % fprintf('lansvd   ');
        lh = lh + 1;
    else
        [U,D,V] = svd(wx, 'econ' );      % fprintf('fullsvd  ');
    end
    D = diag(D);    index = find( D > sct*beta );  % shrinkage operator
    D = diag( D(index) - sct*beta );
    Xnew = U(:,index) * D * V(:,index)';
    SX = ( wx - Xnew )./ beta;          % Update the subgradient of X-part
    out.Rankr(iter) = length( index );        % Record the rank
    svp = out.Rankr(iter);
    if svp < sv
        sv = min( svp + 1, n);
    else
        sv = min( svp + round(0.04*n), n);
    end
    timex = toc;  %timex = cputime - timex;
    
    %%======== Compute the y-subproblem ================================
    tic;  % timey = cputime;
    wy = Y + beta.*SY - alpha.*ey;
    Ynew = sign(wy).* max( abs(wy) - tau*sct*beta, 0 );  % Update Y
    SY = ( wy - Ynew )./ beta;        % Update the subgradient of Y-part
    out.SP(iter) = sum( abs(Ynew(:)) > 1e-2 );
    %     out.SP(iter) = length(find( abs(Ynew(:)) > 1e-2 ));
    timey = toc; % timey = cputime - timey;
    
    %%======== Compute the z-subproblem ================================
    tic; % timez = cputime;
    wz = Z + beta.*SZ - alpha.*ez;
    if strcmp(para.model,'unconstrained')
        factor = mu / ( sct*beta + mu );
    else
        znorm = norm( wz(Omega), 'fro' );
        factor = min( znorm, para.delta)/ znorm;
    end
    wz(Omega) = factor.*(wz(Omega));     Znew = wz;
    SZ = ( wz - Znew )./ beta;        % Update the subgradient of Z-part
    timez = toc; % timez = cputime - timez;
    
    %%======== Update the Lagrangian multiplier ========================
    tic;         % timel = cputime;
    Lam = Lam - alpha.*etxyz;
    timel = toc; % timel = cputime - timel;
    
    time = time + time0 + max([timex,timey,timez,timel]); %Record CPU time;
    timeT = timeT + time0 + timex + timey + timez + timel;
    out.Time(iter) = time;   out.TimeT(iter) = timeT;
    
    out.error = max( norm(Xnew - X,'fro')/(1+norm(Xnew,'fro')), ...
        norm(Ynew - Y,'fro')/(1+norm(Ynew,'fro')) );
    
    X = Xnew;   Y = Ynew;   Z = Znew;   % Update the last iterates
    
    if (detail)    %% Display the details of the procedure
        fprintf('It: %3d cpu: %5.2f  rank: %3d spa: %7d stopic: %4.2e \n',...
            iter,out.Time(iter),out.Rankr(iter),out.SP(iter),out.error);
    end
    
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
end

out.LowRank = X; out.Sparse = Y;  out.M = M;  out.iter = iter;

end
