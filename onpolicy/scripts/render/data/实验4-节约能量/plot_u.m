T = 209;  % 800
t1 = 2:T;
t2 = 2:T;
% plot(t1, png_u(1:162));
% hold on
% plot(t2, rhc_u(1:120));
% legend('PNG', 'RHC');

% t_s = 1:370;
% png_S_ = png_S*(-1);
% rhc_S_ = rhc_S*(-1);
% plot(t_s, png_S_(1:t_s(end)));
% hold on
% plot(t_s, rhc_S_(1:t_s(end)));
% legend('PNG', 'RHC');

plot(t1, png_a4_(2:T));
hold on
plot(t2, rhc_a4_(2:T));
legend('PNG', 'RHC');


