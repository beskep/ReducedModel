clc
clear all
global Measurement1 timestep C K_g_in K_g_out Lin_g Lout_g specific1 ...
    dt To Km_1 Lin_m_1 Lout_m_1 specific1_node M1 ...
    RMSE_C1 RMSE_stack U1 U2 U model1
load 'Initial_matrix_data_point2.mat'

%% Initial Condition

Sink_Temp_in = 20;
Sink_Temp_out = 5;

unit = 30; %분 단위 설정 ex) 1시간 = 60;

dt = unit * 60; % 초단위 설정 unit(=분단위)

before3days = 3 * 24 * 60 / unit; % 안정화를 위한 3일 3일 X 24시간 X 60분 / 30분(분단위)

timestep = before3days + floor(length(data_U1) / unit); % 안정화 구간(3days) + 측정 구간

U1 = zeros(before3days + floor(length(data_U1) / unit), 1); % 실내온도 크기 정의
U2 = zeros(before3days + floor(length(data_U2) / unit), 1); % 실외온도 크기 정의

for i = 1:before3days

    U1(i, 1) = 20; % %안정화를 위한 실내온도 설정 (3일, 20℃)
    U2(i, 1) = 20; % %안정화를 위한 실외온도 설정 (3일, 20℃)

end

for i = 1:floor(length(data_U1) / unit)

    U1(before3days + i, 1) = data_U1(unit * i, 1);
    U2(before3days + i, 1) = data_U2(unit * i, 1);

end

%% Matrix definition

disp('C matrix')
C = eye(length(C_o));

for i = 1:length(C_o)

    C(C_o(i, 1), C_o(i, 2)) = C_o(i, 3);

end

%%
disp('K matrix by Case')
Km_1 = zeros(max(K_o_1(:, 1)), max(K_o_1(:, 2)));
ecount = 0;

for i = 1:length(K_o_1)

    Km_1(K_o_1(i, 1), K_o_1(i, 2)) = K_o_1(i, 3);

    if K_o_1(i, 3) ~= Km_1(K_o_1(i, 1), K_o_1(i, 2)),
        disp('Error')
        ecount = ecount + 1;
    end

end

for i = 1:length(C)

    for j = 1:length(C)

        Km_1(i, j) = Km_1(j, i);

    end

end

Km_1 = -Km_1;

%%
Km_2 = zeros(max(K_o_2(:, 1)), max(K_o_2(:, 2)));
ecount = 0;

for i = 1:length(K_o_2)

    Km_2(K_o_2(i, 1), K_o_2(i, 2)) = K_o_2(i, 3);

    if K_o_2(i, 3) ~= Km_2(K_o_2(i, 1), K_o_2(i, 2)),
        disp('Error')
        ecount = ecount + 1;
    end

end

for i = 1:length(C)

    for j = 1:length(C)

        Km_2(i, j) = Km_2(j, i);

    end

end

Km_2 = -Km_2;

%%
Km_3 = zeros(max(K_o_3(:, 1)), max(K_o_3(:, 2)));
ecount = 0;

for i = 1:length(K_o_3)

    Km_3(K_o_3(i, 1), K_o_3(i, 2)) = K_o_3(i, 3);

    if K_o_3(i, 3) ~= Km_3(K_o_3(i, 1), K_o_3(i, 2)),
        disp('Error')
        ecount = ecount + 1;
    end

end

for i = 1:length(C)

    for j = 1:length(C)

        Km_3(i, j) = Km_3(j, i);

    end

end

Km_3 = -Km_3;

%%
Km_4 = zeros(max(K_o_4(:, 1)), max(K_o_4(:, 2)));
ecount = 0;

for i = 1:length(K_o_4)

    Km_4(K_o_4(i, 1), K_o_4(i, 2)) = K_o_4(i, 3);

    if K_o_4(i, 3) ~= Km_4(K_o_4(i, 1), K_o_4(i, 2)),
        disp('Error')
        ecount = ecount + 1;
    end

end

for i = 1:length(C)

    for j = 1:length(C)

        Km_4(i, j) = Km_4(j, i);

    end

end

Km_4 = -Km_4;

%%
disp('Lin matrix by Case')

Lin_m_1 = zeros(length(C), 1);

for i = 1:length(Lin_o_1)

    Lin_m_1(Lin_o_1(i, 1), 1) = Lin_o_1(i, 2) / Sink_Temp_in;

end

Lin_m_2 = zeros(length(C), 1);

for i = 1:length(Lin_o_2)

    Lin_m_2(Lin_o_2(i, 1), 1) = Lin_o_2(i, 2) / Sink_Temp_in;

end

Lin_m_3 = zeros(length(C), 1);

for i = 1:length(Lin_o_3)

    Lin_m_3(Lin_o_3(i, 1), 1) = Lin_o_3(i, 2) / Sink_Temp_in;

end

Lin_m_4 = zeros(length(C), 1);

for i = 1:length(Lin_o_4)

    Lin_m_4(Lin_o_4(i, 1), 1) = Lin_o_4(i, 2) / Sink_Temp_in;

end

%%
disp('Lout matrix by Case')

Lout_m_1 = zeros(length(C), 1);

for i = 1:length(Lout_o_1)

    Lout_m_1(Lout_o_1(i, 1), 1) = Lout_o_1(i, 2) / Sink_Temp_out;

end

Lout_m_2 = zeros(length(C), 1);

for i = 1:length(Lout_o_2)

    Lout_m_2(Lout_o_2(i, 1), 1) = Lout_o_2(i, 2) / Sink_Temp_out;

end

Lout_m_3 = zeros(length(C), 1);

for i = 1:length(Lout_o_3)

    Lout_m_3(Lout_o_3(i, 1), 1) = Lout_o_3(i, 2) / Sink_Temp_out;

end

Lout_m_4 = zeros(length(C), 1);

for i = 1:length(Lout_o_4)

    Lout_m_4(Lout_o_4(i, 1), 1) = Lout_o_4(i, 2) / Sink_Temp_out;

end

%% Specific surface avr temp

specific1 = zeros(length(C), 1);

for i = 1:length(specific1_node)

    specific1(specific1_node(i, 1), 1) = specific1_node(i, 2);

end

%% K 가중치 (실내측)

K_g_in = Km_4 - Km_1; %Global--> K_g_in

%% K 가중치 (실외측)
K_g_out = Km_3 - Km_1; %Global--> K_g_out

%% L 가중치 (실내측)

Lin_g = Lin_m_2 - Lin_m_1; %Global--> Lin_g

%% L 가중치 (실외측)

Lout_g = Lout_m_2 - Lout_m_1; %Global--> Lout_g

%% 초기조건 추가필요

To = ones(length(C), 1) * 10;
M1 = zeros(1, timestep);

%% Particle Swarm Optimization

Bounds = [1, 10; 5, 30];

ParaName = {'Hin', 'Hout'};
BoundInit = zeros(1, 2)

tic

for i = 1:2
    BoundInit(i) = mean(Bounds(i, :));
end

for i = 1:2
    eval(sprintf('%s=%f;', ParaName{i}, BoundInit(i)))
end

Para = [Hin Hout];

lb = Bounds(:, 1)';
ub = Bounds(:, 2)';

options = optimoptions('particleswarm', 'Swarmsize', 500, 'HybridFcn', @fmincon, 'MaxTime', 100, 'ObjectiveLimit', 0);

f = @FUNCTION_code_point2; % 여기서부터 FUNCTION_code_point2 으로 넘어감

[fPara, fval, exitflag, output] = particleswarm(f, 2, lb, ub, options)
Hin = fPara(1);
Hout = fPara(2);

ParaFinal = [Hin Hout];
toc
