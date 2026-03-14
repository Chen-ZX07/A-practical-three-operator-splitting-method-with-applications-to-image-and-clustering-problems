D = B;
b = g;

mu1 = 0.1;
mu2 = 0.1;
mubar = 5*(max(max(D))*8+1)/4;

x = zeros(MN,1);
y = zeros(2*MN,1);
u = zeros(2*MN,1);

theta = 1;
a = 0;
bk = 0;
maxiter = 500;
tol = 1e-6;
% define the objective function:
f = @(x) norm(B*x,1)+lambda*norm(x-g,1);
f_end=f(x_cvx);
tic;
for i=1:maxiter
    %a=x-b;
    ubar = u+(D*x-y)/mubar;
    xbar = max(x-(a+D'*ubar)/mu1,1e-6);
    xkk=x-xbar;
    ukk=u-ubar;
    Dxy = D*xkk-y;
    Du = mubar*ukk + Dxy;
    a1=norm(xkk,2)^2;
    a2=norm(y,2)^2;
    phi = mu1*a1+mu2*a2+Du'*ukk;
    psi = a1 + a2+ norm(Du,2)^2;
    gamma = theta * phi / psi;
    g1 = x+(a-gamma*xkk)/mu1;
    xk = sign(g1-b).*max(abs(g1-b)-1/mu1,1e-6)+b;
    %xk=(g1+b)/(mu1+1);
    g2 = (1-gamma/mu2)*y+bk/mu2;
    yk = max(g2-1/mu2,1e-6)+min(g2+1/mu2,1e-6);
    u = u-gamma*Du;
    a = mu1 * (x-xk) + a - gamma * xkk;
    %bk = mu2 * (y-yk) + bk - gamma * y;
    bk = (mu2 - gamma)* y- mu2 * yk + bk;
    f_our(i) = abs(f(xk)-f_end)/f_end;
    if f_our(i)<tol
        break;
    end
    x = xk;
end
time_our = toc;

xour=reshape(x, 1024, 1024);
imshow(xour)
our_psnr= psnr(f0,xour)

