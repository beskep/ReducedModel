function [RMSE_C] = FUNCTION_code_point2(Para)
    global Measurement1 timestep C K_g_in K_g_out Lin_g Lout_g specific1 ...
        dt To Km_1 Lin_m_1 Lout_m_1 specific1_node M1 ...
        RMSE_C1 RMSE_stack U1 U2 U model1

    Hin = Para(1);
    Hout = Para(2);

    %% K matrix definition

    K = Km_1 + (K_g_in * (Hin - 1) + K_g_out * (Hout - 1));

    %% L matrix definition

    Lin = Lin_m_1 + (Lin_g * (Hin - 1));

    Lout = Lout_m_1 + (Lout_g * (Hout - 1));

    L = zeros(length(C), 2);

    for i = 1:length(C)

        L(i, 1) = Lin(i, 1);
        L(i, 2) = Lout(i, 1);

    end

    %% Reduction

    A = inv(C) * K;
    B = inv(C) * L;
    J = [specific1' / length(specific1_node); ];
    D = 0;

    sys = ss(A, B, J, D);
    Order = 10;

    model1 = balred(sys, Order);
    Ar = model1.A;
    Br = model1.B;
    Cr = model1.C;
    Dr = model1.D;

    Red1 = inv(eye(length(Ar)) - dt * Ar);
    Red2 = dt * Red1 * Br;

    Xn = zeros(length(Ar), 1);
    Xo = zeros(length(Ar), 1);

    for i = 1:timestep

        Xn = Red1 * Xo + Red2 * [U1(i, 1); U2(i, 1)];
        Y = Cr * Xn + Dr * [U1(i, 1); U2(i, 1)];

        M1(:, i) = Y(1, 1);

        Xo = Xn;

    end

    RMSE_C1 = sqrt(mean((Measurement1(1, 1) - M1(1, timestep)).^2));

    % RMSE_tot = [RMSE_C1; RMSE_C2; RMSE_C3;];

    RMSE_C = RMSE_C1

    RMSE_stack = [RMSE_stack; RMSE_C];

end

%% Calculation 2 (Modifying Matrix)

% S1=inv(C)*dt;
% S2=inv(eye(length(C))-S1*K);
%
% for i=1:timestep
%
% %     T=S2*(To+S1*(Lin*20)+(Lout*5));
%     T=S2*(To+S1*(Lin*20+Lout*5));
%     Y1(i)=specific1'/length(specific1_node)*T;
%     Y2(i)=specific2'/length(specific2_node)*T;
%     Y3(i)=specific3'/length(specific3_node)*T;
%
%     To=T;
%     M1(:,i)=Y1(i);
%     M2(:,i)=Y2(i);
%     M3(:,i)=Y3(i);
%
% end
%
%
%
% RMSE_C1=sqrt(mean((Measurement1(1,timestep) - M1(1,timestep)).^2));
% RMSE_C2=sqrt(mean((Measurement2(1,timestep) - M2(1,timestep)).^2));
% RMSE_C3=sqrt(mean((Measurement3(1,timestep) - M3(1,timestep)).^2));
%
% RMSE_tot = [RMSE_C1; RMSE_C2; RMSE_C3;];
%
% RMSE_C = max(RMSE_tot);
%
% RMSE_stack = [RMSE_stack; RMSE_C]
% % plot(1:length(RMSE_stack),RMSE_stack);
%
% end
