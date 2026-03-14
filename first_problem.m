clc; clear; close all;

% =================== test the list of m ===================
m_list = [1000 2000 3000 5000];


results = struct();

for idx = 1:length(m_list)

    % =================== 基本设置 ===================
    m    = m_list(idx);
    kmax = 100;
    gap  = 5;   % 记录/绘图的间隔

    CPU = []; It = []; jingdu = [];
    modk1 = []; modklog1 = []; modk11 = [];
    modk2 = []; modklog2 = []; modk22 = [];
    modk3 = []; modklog3 = []; modk33 = [];
    modk4 = []; modklog4 = []; modk44 = [];
    modk5 = []; modklog5 = []; modk55 = [];
    modk6 = []; modklog6 = []; modk66 = [];
    modk7 = []; modklog7 = []; modk77 = [];

    %%%%%%  0 ∈ C(x) + A(x) + Q*B(Qx - q)
    % 真实解 x* = e1, 维数 m, Q -- (m+1)*m
    h = 1/(m+1);
    e = ones(m,1);
    D = spdiags([(-1-h)*e (4+2*h)*e -1*e], -1:1, m, m);
    e1 = [1; zeros(m-1,1)];

    d = D*e1;
    M = 0.5*(D + D');   % C(x) = M x - d
    A = 0.5*(D - D');   % 斜对称
    N = A;
    B = M;              % 为兼容之前代码

    I = eye(m);
    Q = [I; (-1/m)*ones(1,m)];         % (m+1) x m
    q = [zeros(m,1); -1/m];            % (m+1) x 1

    %% ===================== 1) Algorithm 2 (Ours) =====================
    alpha = 6;
    alpha_hat = 5*norm(Q,1)*norm(Q,'inf')/(4*alpha);
    theta = 1;
    xk = zeros(m,1);
    uk = zeros(m+1,1);
    k  = 0;
    etime1 = 0;
    while k <= kmax
        tic;
        aA   = A*xk;
        Sk   = uk + (Q*xk - q)/alpha_hat;
        ubar = Sk - max(0,Sk);                                % 投到非正正交
        xbar = (alpha*I + B) \ (alpha*xk - aA - Q'*ubar + d); % resolvent
        xxb  = xk - xbar;
        uub  = uk - ubar;
        aq2  = alpha_hat*uub + Q*xxb;

        g1   = sum(alpha.*xxb.^2) + aq2'*uub;
        g2   = sum(xxb.^2) + sum(aq2.^2);
        gamma = theta * g1 / g2;

        xk = (alpha*I + A) \ (alpha*xk + aA - gamma*xxb);
        uk = uk - gamma*aq2;

        et = toc; etime1 = etime1 + et;
        if mod(k,gap)==0
            modk1    = [modk1, etime1];
            modk11   = [modk11, k];
            modklog1 = [modklog1, log10(norm(xk - e1))];
        end
        k = k + 1;
    end
    CPU(end+1)    = etime1;
    jingdu(end+1) = norm(xk - e1);
    It(end+1)     = k;

    %% ===================== 2) Algorithm 1 (Dong) =====================
    theta = 1.2;
    xk = zeros(m,1);
    uk = zeros(m+1,1);
    xnplus1 = uk;
    ak = A*xk;         % 记 a_k = A x_k

    k = 0; etime2 = 0;
    alpha1 = 1; alpha2 = 1;
    beta1  = norm(Q,1)*norm(Q,inf)/(4*alpha1);
    beta2  = 1/(4*alpha2);
    beta   = 2*(beta1+beta2) + 1e-9*min(beta1,beta2);

    while k <= kmax
        tic;
        bar_uk       = uk + (Q*xk - q - xnplus1)/beta;
        bar_xk       = (alpha1*I + B) \ (d + alpha1*xk - ak - Q'*bar_uk);
        bar_xnplus1  = max(0, alpha2*xnplus1 + bar_uk);

        xkbarxk      = xk - bar_xk;
        xkbarxknp1   = xnplus1 - bar_xnplus1;
        sumdot       = -Q*bar_xk + q + bar_xnplus1;

        Phik = alpha1*sum(xkbarxk.^2) + alpha2*sum(xkbarxknp1.^2) + sumdot'*(uk - bar_uk);
        phik = sum(xkbarxk.^2) + sum(xkbarxknp1.^2) + sum(sumdot.^2);
        gammak = theta * Phik / phik;

        xkplus1   = (alpha1*I + A) \ (alpha1*xk + ak - gammak*xkbarxk);
        xnPLUS1   = xnplus1 - (gammak/alpha2)*xkbarxknp1;
        ukplus1   = uk - gammak*sumdot;
        akplus1   = alpha1*(xk - xkplus1) + ak - gammak*xkbarxk;

        et = toc; etime2 = etime2 + et;
        if mod(k,gap)==0
            modk2    = [modk2, etime2];
            modk22   = [modk22, k];
            modklog2 = [modklog2, log10(norm(xk - e1))];
        end

        k = k + 1;
        xk = xkplus1;  xnplus1 = xnPLUS1;  uk = ukplus1;  ak = akplus1;
    end
    CPU(end+1)    = etime2;
    jingdu(end+1) = norm(xk - e1);
    It(end+1)     = k;

    %% ===================== 3) DL-Pseudo Splitting ====================
    theta_p = 1.2;
    alpha1p = 1;
    alpha2p = 1;
    beta_p  = 5;

    x = zeros(m,1);
    y = zeros(m+1,1);
    u = zeros(m+1,1);
    k = 0; etime5 = 0;

    while k <= kmax
        tic;
        ubar = u - (y - Q*x + q)/beta_p;
        xbar = (alpha1p*I + B) \ (alpha1p*x - Q'*ubar + d);
        ybar = max(0, y + (ubar/alpha2p));
        dx   = x - xbar;
        dy   = y - ybar;
        r    = ybar - Q*xbar + q;

        Phi  = alpha1p*sum(dx.^2) + alpha2p*sum(dy.^2) + r'*(u - ubar);
        Psi  = sum((alpha1p*dx).^2) + sum((alpha2p*dy).^2) + sum(r.^2);
        gamma = theta_p * (Phi / Psi);

        x = x - gamma*alpha1p*dx;
        y = y - gamma*alpha2p*dy;
        u = u - gamma*r;

        et = toc; etime5 = etime5 + et;
        if mod(k,gap)==0
            modk5    = [modk5, etime5];
            modk55   = [modk55, k];
            modklog5 = [modklog5, log10(norm(x - e1))];
        end
        k = k + 1;
    end
    CPU(end+1)    = etime5;
    jingdu(end+1) = norm(x - e1);
    It(end+1)     = k;

    %% ===================== 4) Vũ–Condat Splitting =====================
    L1 = eye(m); r1 = zeros(m,1);
    L2 = Q;      r2 = q;

    xstar = e1;
    xk = zeros(m,1);
    v1 = zeros(m,1);
    v2 = zeros(m+1,1);
    tao = 0.5;
    lambda = 1.4;
    sigma1 = 1.6;
    sigma2 = 1.6;

    if sigma1*norm(L1)^2 + sigma2*norm(L2)^2 >= 2/tao
        warning('Vu Splitting: sigma1*||L1||^2 + sigma2*||L2||^2 >= 2/tao');
    end

    B1 = A;
    k = 0; etime3 = 0;

    while k <= kmax
        tic;
        nu_vec = xk - 0.5*tao*(L1'*v1 + L2'*v2);
        p  = (I + (tao/2)*B) \ (nu_vec + (tao/2)*d);
        yv = 2*p - xk;
        xk = xk + lambda*(p - xk);

        u1 = v1 + sigma1*(L1*yv - r1);
        mu1 = (I + sigma1*(B1\I)) \ u1;
        v1  = v1 + lambda*(mu1 - v1);

        u2 = v2 + sigma2*(L2*yv - r2);
        mu2 = u2 - sigma2*max(0, u2/sigma2);
        v2  = v2 + lambda*(mu2 - v2);

        et = toc; etime3 = etime3 + et;
        if mod(k,gap)==0
            modk3    = [modk3, etime3];
            modk33   = [modk33, k];
            modklog3 = [modklog3, log10(norm(xk - xstar))];
        end
        k = k + 1;
    end
    CPU(end+1)    = etime3;
    jingdu(end+1) = norm(xk - xstar);
    It(end+1)     = k;

    %% ===================== 5) JE Splitting ===========================
    G1 = Q; G2 = eye(m); G3 = G2;
    z  = zeros(m,1);
    w1 = zeros(m+1,1);
    w2 = B*z - d;
    w3 = -G1'*w1 - G2'*w2;
    x1 = zeros(m+1,1);
    x2 = zeros(m,1);
    x3 = zeros(m,1);
    xstar = e1;

    alphaJE = [0.9, 0.9, 0.9];
    rhoJE   = [1,1,1]';
    gammaJE = 10;

    etime4 = 0; k = 0;
    while k <= kmax
        tic;
        t1 = (1 - alphaJE(1))*x1 + alphaJE(1)*G1*z + rhoJE(1)*w1;
        x1 = max(q, t1);
        y1 = (1/rhoJE(1))*(t1 - x1);

        t2 = (1 - alphaJE(2))*x2 + alphaJE(2)*G2*z + rhoJE(2)*w2;
        x2 = (eye(m) + 0.5*rhoJE(2)*(D + D')) \ (t2 + rhoJE(2)*d);
        y2 = (1/rhoJE(2))*(t2 - x2);

        t3 = (1 - alphaJE(3))*x3 + alphaJE(3)*G3*z + rhoJE(3)*w3;
        x3 = (eye(m) + 0.5*rhoJE(3)*(D - D')) \ t3;
        y3 = (1/rhoJE(3))*(t3 - x3);

        u1 = x1 - G1*x3;
        u2 = x2 - G2*x3;
        u  = [u1; u2];
        v  = G1'*y1 + G2'*y2 + y3;

        piJE = norm(u)^2 + (1/gammaJE)*norm(v)^2;
        if piJE > 0
            phiJE  = z'*v + w1'*u1 + w2'*u2 - (x1'*y1 + x2'*y2 + x3'*y3);
            taoJE  = (1/piJE)*max(0, phiJE);
        else
            taoJE  = 0;
        end

        et = toc; etime4 = etime4 + et;
        if mod(k,gap)==0
            modk4    = [modk4, etime4];
            modk44   = [modk44, k];
            modklog4 = [modklog4, log10(norm(z - xstar))];
        end

        z  = z  - (1/gammaJE)*taoJE*v;
        w1 = w1 - taoJE*u1;
        w2 = w2 - taoJE*u2;
        w3 = -G1'*w1 - G2'*w2;
        k = k + 1;
    end
    CPU(end+1)    = etime4;
    jingdu(end+1) = norm(z - xstar);
    It(end+1)     = k;

    %% ===================== 6) Algorithm 6 (tang2022, improved) =======
    eps_  = 0.4;
    nu    = 0.85;
    tau   = 0.30;               % 允许更大的 τ
    sigma = 0.20;               % 稍微减小 sigma
    omega = 1.25;               % 轻微超松弛

    xk = zeros(m,1);
    yk = zeros(m+1,1);

    etime6 = 0;
    for k = 0:kmax
        tic;
        % Step 1
        z1 = xk - (sigma/2) * (Q' * yk);
        p1 = (I + sigma*A) \ z1;

        % Step 2
        S  = yk - (tau/2) * ( Q * (2*p1 - xk) );
        qk = max(q, S);

        % Step 3
        z2 = (2*p1 - xk) - (sigma/2) * ( Q' * (2*qk - yk) );
        p2 = (I + sigma*B) \ ( z2 + sigma*d );

        % Update
        x_new = xk + omega*(p2 - p1);
        y_new = yk - omega*(qk - q);

        x_primal = p2;   % 当前原始解

        et = toc; etime6 = etime6 + et;
        if mod(k,gap)==0
            modk6    = [modk6, etime6];
            modk66   = [modk66, k];
            modklog6 = [modklog6, log10(norm(x_primal - e1))];
        end

        xk = x_new;
        yk = y_new;
    end
    CPU(end+1)    = etime6;
    jingdu(end+1) = norm(x_primal - e1);
    It(end+1)     = k;

    %% ===================== 7) Algorithm 3.1 (VLG 2024) ===============
    xk = zeros(m,1);
    uk = zeros(m+1,1);
    alphak = 0.6;
    epsA   = 1e-9;
    t1     = 1;
    betak  = (2*t1*epsA - epsA^2 + alphak^2*norm(Q)^2) / (2*(2*t1 - epsA)*alphak);
    betak  = max(betak, alphak);

    k = 0; etime7 = 0;

    while k <= kmax
        tic;
        xk_bar = (I + alphak*B) \ (xk - alphak*(N*xk + Q'*uk) + alphak*d);
        yk     = betak*uk + Q*xk_bar - q;
        uk_bar = (1/betak)*yk - (1/betak)*max(zeros(m+1,1), yk);

        Qu   = Q'*(uk - uk_bar);
        dk_x = (1/alphak)*(xk - xk_bar) - N*(xk - xk_bar) - Qu;
        dk_u = betak*(uk - uk_bar);

        if norm(xk - xk_bar)==0 && norm(uk - uk_bar)==0
            xk_plus1 = xk;
            uk_plus1 = uk;
        else
            gamma = (dot(xk - xk_bar, dk_x) + dot(uk - uk_bar, dk_u)) / ...
                    (norm(dk_x)^2 + norm(dk_u)^2);
            xk_plus1 = xk - gamma*dk_x;
            uk_plus1 = uk - gamma*dk_u;
        end

        alpha_plus1 = alphak;   % 步长保持不变

        et = toc; etime7 = etime7 + et;
        if mod(k,gap)==0
            modk7    = [modk7, etime7];
            modk77   = [modk77, k];
            modklog7 = [modklog7, log10(norm(xk - e1))];
        end

        k      = k + 1;
        xk     = xk_plus1;
        uk     = uk_plus1;
        alphak = alpha_plus1;
    end
    CPU(end+1)    = etime7;
    jingdu(end+1) = norm(xk-e1);
    It(end+1)     = k;

    %% ===== 把本次 m 的结果存到 results 里，方便后面统一画图 =====
    results(idx).m    = m;
    results(idx).modk1 = modk1;  results(idx).log1 = modklog1;
    results(idx).modk2 = modk2;  results(idx).log2 = modklog2;
    results(idx).modk3 = modk3;  results(idx).log3 = modklog3;
    results(idx).modk4 = modk4;  results(idx).log4 = modklog4;
    results(idx).modk5 = modk5;  results(idx).log5 = modklog5;
    results(idx).modk6 = modk6;  results(idx).log6 = modklog6;
    results(idx).modk7 = modk7;  results(idx).log7 = modklog7;

end

%% ====================== 四宫格绘图并保存 ========================
% figure('Color','w','Position',[100 100 900 700]);
% 
% for idx = 1:length(m_list)
%     subplot(2,2,idx); hold on; grid on; box on;
% 
%     m = results(idx).m;
%     R = results(idx);
% 
%     % 按照：Vu, JE, DL, Dong, Ours, VLG2024, tang2022 的顺序画线
%     plot(R.modk3,R.log3,'-s','LineWidth',1.5,'MarkerSize',4); % Vu
%     plot(R.modk4,R.log4,'-+','LineWidth',1.5,'MarkerSize',4); % JE
%     % plot(R.modk5,R.log5,'-^','LineWidth',1.5,'MarkerSize',4); % DL
%     plot(R.modk2,R.log2,'-*','LineWidth',1.5,'MarkerSize',4); % Dong
%     plot(R.modk1,R.log1,'-o','LineWidth',1.5,'MarkerSize',4); % Ours
%     plot(R.modk7,R.log7,'-d','LineWidth',1.5,'MarkerSize',4); % VLG2024
%     plot(R.modk6,R.log6,'-v','LineWidth',1.5,'MarkerSize',4); % tang2022
% 
%     title(sprintf('m = %d', m));
%     xlabel('Time (seconds)');
%     ylabel('log_{10}(||x^k - x^*||)','Interpreter','tex');
%     set(gca,'FontName','Times New Roman','FontSize',11,'LineWidth',1);
% 
%     subplot(2,2,4)
%         legend('Vu Splitting','JE Splitting', ...
%                'Algorithm 1 (Dong)','Algorithm 2 (Ours)', ...
%                'VLG 2024','tang2022',...
%                'Location','southwest');
% 
% 
% end
% sgtitle('The first test problem, m = 1000, 2000, 3000, 5000', ...
%         'FontSize',14);
%% 绘制第一组测试问题的收敛曲线
figure('Color','w','Position',[100 100 900 700]);

% 使用 tiledlayout 管理 2x2 子图
t = tiledlayout(2,2, ...
    'Padding','compact', ...      % 子图与边缘留白更紧凑
    'TileSpacing','compact');     % 子图之间间距更小

for idx = 1:length(m_list)
    % 选择当前子图
    nexttile; 
    hold on; grid on; box on;

    m = results(idx).m;
    R = results(idx);

    % 按顺序绘制各算法的折线
    % Vu Splitting
    plot(R.modk3, R.log3, '-s', 'LineWidth', 1.5, 'MarkerSize', 4);
    % JE Splitting
    plot(R.modk4, R.log4, '-+', 'LineWidth', 1.5, 'MarkerSize', 4);
    % Algorithm 1 (Dong)
    plot(R.modk2, R.log2, '-*', 'LineWidth', 1.5, 'MarkerSize', 4);
    % Algorithm 2 (Ours)
    plot(R.modk1, R.log1, '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    % VLG 2024
    plot(R.modk7, R.log7, '-d', 'LineWidth', 1.5, 'MarkerSize', 4);
    % tang2022
    plot(R.modk6, R.log6, '-v', 'LineWidth', 1.5, 'MarkerSize', 4);

    % 子图标题、坐标轴标签
    title(sprintf('m = %d', m), ...
        'FontName', 'Times New Roman', 'FontSize', 11);
    xlabel('Time (seconds)', ...
        'FontName', 'Times New Roman', 'FontSize', 11);
    ylabel('log_{10}(||x^k - x^*||)', ...
        'Interpreter', 'tex', ...
        'FontName', 'Times New Roman', 'FontSize', 11);

    set(gca, 'FontName','Times New Roman', ...
             'FontSize',11, ...
             'LineWidth',1);
end

% ------------ 顶部统一图例 ------------
lg = legend({'Vu Splitting','JE Splitting','Algorithm 1','Algorithm 2', ...
             'VLG 2024', 'Tang2022'}, ...
            'Orientation','horizontal', ...   % 横向排布
            'NumColumns', 6, ...             % 每行放 3 个
            'Box','off', ...
            'FontName','Times New Roman', ...
            'FontSize',11);

% 让图例占据整体布局的最上方
lg.Layout.Tile = 'north';

% ------------ 整体标题 ------------
title(t, 'The first test problem, m = 1000, 2000, 3000, 5000', ...
      'FontName','Times New Roman', ...
      'FontSize',14, ...
      'FontWeight','normal');


% 保存四宫格
saveas(gcf,'first_test_multi_m.fig');
saveas(gcf,'first_test_multi_m.png');
