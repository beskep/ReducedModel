clear all
clc

function [YY] = reduced_model_code(Y)

    load('Reduction_model_matfile.mat')

    tic
    %% Matrix 정의

    %% diagonal matrix 계산시간을 줄이기 위해 단위 행렬 C1을 생성함
    C1 = eye(length(C));

    % sparse matrix -> dense matrix
    for i = 1:length(C)
        C1(C(i, 1), C(i, 2)) = C(i, 3);

    end

    %% Marix K

    %% Kcomb 반복문 계산시간을 줄이기 위해 다음과 같이 for문 전 미리 정의함
    Kcomb = zeros(max(K(:, 1)), max(K(:, 2)));

    ecount = 0;

    % sparse matrix -> dense matrix
    for i = 1:length(K)

        Kcomb(K(i, 1), K(i, 2)) = K(i, 3);

        if K(i, 3) ~= Kcomb(K(i, 1), K(i, 2)),
            disp('ERROR')
            %% data K의 3행의 값이 Kcomb에 대응되는 값과 같지 않을 경우
            %% Error (동일데이터가 들어가도록 확인하는 용도)

            % ecount 사용되지 않음
            ecount = ecount + 1;
        end

    end

    %% Matrix Kcomb symmetry
    disp('a')

    for i = 1:length(C)

        for j = 1:length(C)

            Kcomb(i, j) = Kcomb(j, i);

        end

    end

    disp('b')
    Kcomb = -Kcomb;

    %% Matrix Lin

    L1 = zeros(length(C), 1);

    for i = 1:length(Lin)
        L1(Lin(i, 1), 1) = Lin(i, 2) / 20;
    end

    %% Matrix Lout
    L2 = zeros(length(C), 1);

    for i = 1:length(Lout)
        L2(Lout(i, 1), 1) = Lout(i, 2) / 10;
    end

    %% 특정부위 노드 위치 (Specific 1)
    specific1_1 = zeros(length(C1), 1);

    for i = 1:length(specific1)
        specific1_1(specific1(i, 1), 1) = specific1(i, 2);
    end

    %% 특정부위 노드 위치 (Specific 2)

    specific2_1 = zeros(length(C1), 1);

    for i = 1:length(specific2)
        specific2_1(specific2(i, 1), 1) = specific2(i, 2);
    end

    %% 특정부위 노드 위치 (Specific 3)
    specific3_1 = zeros(length(C1), 1);

    for i = 1:length(specific3)
        specific3_1(specific3(i, 1), 1) = specific3(i, 2);

    end

    %% 데이터 체크

    check1 = sum(Kcomb(1, :));
    check2 = sum(L1(1, :));
    check3 = sum(L2(1, :));

    if check1 ~= -1 * (check2 + check3)
        disp('okay')
    end

    disp('c')

    dt = 3600;
    timestep = 192;

    %% Reduction
    L3 = zeros(length(C1), 2);

    for i = 1:length(C1)
        L3(i, 1) = L1(i, 1);
        L3(i, 2) = L2(i, 1);
    end

    A = inv(C1) * Kcomb;
    B = inv(C1) * L3;
    J = [specific1_1' / length(specific1);
        specific2_1' / length(specific2);
        specific3_1' / length(specific3); ];
    D = 0;

    Order = 10;
    sys = ss(A, B, J, D);

    test1 = balred(sys, Order);
    Ar = test1.A;
    Br = test1.B;
    Cr = test1.C;
    Dr = test1.D;

    Red1 = inv(eye(Order) - dt * Ar);
    Red2 = dt * Red1 * Br;

    Xn = zeros(Order, 1);
    Xo = zeros(Order, 1);

    for i = 1:timestep
        U1 = 20 + sin(pi * 0.5 * i / 6);
        U2 = 10 + 5 * sin(pi * 0.5 * i / 6);
        U = [U1; U2];

        Xn = Red1 * Xo + Red2 * U;
        Y = Cr * Xn + Dr * U;
        YY(:, i) = Y;

        Xo = Xn;
    end

    YY = YY';
    toc

end
