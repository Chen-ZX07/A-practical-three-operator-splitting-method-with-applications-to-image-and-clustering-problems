function  out = admm(m,n,Omega,b,para,stop)

flag = exist('para');
if(flag && isfield(para,'detail'))
    detail = pars.detail;
else
    detail = 0;
end

lambda = 0.15;
bita = 0.2;

D = zeros(m,n);   D(Omega) = b; 
A_hat = zeros(m,n);   E_hat = A_hat;    F_hat = A_hat; 
Y = D;
norm_two = lansvd(Y, 1, 'L');
norm_inf = norm( Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;
rho = 1.2;
mu = 1.25/norm_two;  % Regularized parameters
mu_bar = mu * 1e7;
for iter = 1:stop.Max 
    iter
    temp_T = D - A_hat -F_hat+ (1/mu)*Y;
    Ek_hat = max(temp_T - lambda/mu, 0);
    Ek_hat = Ek_hat+min(temp_T + lambda/mu, 0);
    Fk_hat = (mu*(D-A_hat-Ek_hat)+Y)/(bita+mu);
    temp = D - Ek_hat-Fk_hat + (1/mu)*Y;
    [U, S, V] = svd(temp,'econ');
    diagS = diag( S );
    diagS = max( abs( diagS ) - 1/mu,0);
    Ak_hat = U * diag( diagS ) * V';     
    Z = D - Ak_hat - Ek_hat-Fk_hat;
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
    

    out.error = max( norm(Ak_hat - A_hat,'fro')/(1+norm(Ak_hat,'fro')), ...
        norm(Ek_hat - E_hat,'fro')/(1+norm(Ek_hat,'fro')) + norm(Fk_hat - F_hat,'fro')/(1+norm(Fk_hat,'fro')));
    out.LowRank = Ak_hat;
    out.Sparse = Ek_hat;
    out.M = D;
    out.iter = iter;
    A_hat=Ak_hat;
    E_hat=Ek_hat;
    F_hat=Fk_hat;

    out.Rankr(iter) = rank(A_hat);
    out.SP(iter) = sum( abs(E_hat(:)) > 1e-2 );
    
    [~,VV,~] = svd(A_hat, 'econ');
    out.obj(iter) = sum(diag(VV)) + lambda*sum(abs(E_hat(:))) + 0.5*bita*norm(F_hat(Omega),'fro')^2;
    if strcmp(stop.rule,'TOL')
        if out.error <= stop.eps || iter >= stop.Max
            out.LowRank = A_hat;   out.Sparse = E_hat;    out.M = D;
            out.iter = iter;       out.noise  = F_hat;
            return;
        end
    end
    
    if (detail)    %% Display the details of the procedure
        fprintf('It: %3d cpu: %5.2f  rank: %3d spa: %7d stopic: %4.2e \n',...
            iter,out.Rankr(iter),out.SP(iter),out.error);
    end

end
