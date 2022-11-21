%% Plot ACFs

t1 = tiledlayout(4,4,'TileSpacing','tight');
t2 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t3 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t4 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t5 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t6 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t7 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t8 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t9 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t10 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t11 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t12 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t13 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t14 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t15 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t16 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t17 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');

t3.Layout.Tile = 2;
t4.Layout.Tile = 3;
t5.Layout.Tile = 4;
t6.Layout.Tile = 5;
t7.Layout.Tile = 6;
t8.Layout.Tile = 7;
t9.Layout.Tile = 8;
t10.Layout.Tile = 9;
t11.Layout.Tile = 10;
t12.Layout.Tile = 11;
t13.Layout.Tile = 12;
t14.Layout.Tile = 13;
t15.Layout.Tile = 14;
t16.Layout.Tile = 15;
t17.Layout.Tile = 16;

%title(tPt,'Commodity Prices')
xlabel(t1,'Lags', 'FontSize',16)
ylabel(t1,'\rho_k', 'FontWeight','bold','FontSize',16)

ub = ones(67,1)*(1.96*(1/sqrt(1331)));
lb = ones(67,1)*(-1.96*(1/sqrt(1331)));
x=1:1:67;

nexttile(t2)
bar(x,resSumStats.ACF_rt(:,1));
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t2)
bar(resSumStats.ACF_RV(:,1))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t2,'Crude Oil', 'FontWeight','bold','FontSize',15)

nexttile(t3)
bar(resSumStats.ACF_rt(:,2))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t3)
bar(resSumStats.ACF_RV(:,2))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t3,'Brent Oil', 'FontWeight','bold','FontSize',15)

nexttile(t4)
bar(resSumStats.ACF_rt(:,3))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t4)
bar(resSumStats.ACF_RV(:,3))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t4,'Gasoline', 'FontWeight','bold','FontSize',15)

nexttile(t5)
bar(resSumStats.ACF_rt(:,4))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t5)
bar(resSumStats.ACF_RV(:,4))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t5,'Heating Oil', 'FontWeight','bold','FontSize',15)


nexttile(t6)
bar(resSumStats.ACF_rt(:,5))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t6)
bar(resSumStats.ACF_RV(:,5))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t6,'Gold', 'FontWeight','bold','FontSize',15)

nexttile(t7)
bar(resSumStats.ACF_rt(:,6))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t7)
bar(resSumStats.ACF_RV(:,6))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t7,'Silver', 'FontWeight','bold','FontSize',15)

nexttile(t8)
bar(resSumStats.ACF_rt(:,7))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t8)
bar(resSumStats.ACF_RV(:,7))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t8,'Platinum', 'FontWeight','bold','FontSize',15)

nexttile(t9)
bar(resSumStats.ACF_rt(:,8))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t9)
bar(resSumStats.ACF_RV(:,8))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t9,'Palladium', 'FontWeight','bold','FontSize',15)


nexttile(t10)
bar(resSumStats.ACF_rt(:,9))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t10)
bar(resSumStats.ACF_RV(:,9))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t10,'Copper', 'FontWeight','bold','FontSize',15)

nexttile(t11)
bar(resSumStats.ACF_rt(:,10))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t11)
bar(resSumStats.ACF_RV(:,10))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t11,'Aluminium', 'FontWeight','bold','FontSize',15)

nexttile(t12)
bar(resSumStats.ACF_rt(:,11))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t12)
bar(resSumStats.ACF_RV(:,11))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t12,'Lead', 'FontWeight','bold','FontSize',15)

nexttile(t13)
bar(resSumStats.ACF_rt(:,12))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t13)
bar(resSumStats.ACF_RV(:,12))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t13,'Zinc', 'FontWeight','bold','FontSize',15)


nexttile(t14)
bar(resSumStats.ACF_rt(:,13))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t14)
bar(resSumStats.ACF_RV(:,13))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t14,'Corn', 'FontWeight','bold','FontSize',15)

nexttile(t15)
bar(resSumStats.ACF_rt(:,14))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t15)
bar(resSumStats.ACF_RV(:,14))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t15,'Soybean', 'FontWeight','bold','FontSize',15)

nexttile(t16)
bar(resSumStats.ACF_rt(:,15))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t16)
bar(resSumStats.ACF_RV(:,15))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t16,'Soybean Oil', 'FontWeight','bold','FontSize',15)

nexttile(t17)
bar(resSumStats.ACF_rt(:,16))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
nexttile(t17)
bar(resSumStats.ACF_RV(:,16))
hold on
plot(x,ub,'--r',x,lb,'--r');
hold off
title(t17,'Oats', 'FontWeight','bold','FontSize',15)

% Keep clean the workspace
clearvars t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12 t13 t14 t15 t16 t17 lb ub x

%% Density estimates

tKdens = tiledlayout(4,4,'TileSpacing','tight')
%title(tKdens,'Distribution estimates')
xlabel(tKdens,'y')
ylabel(tKdens,'f(y)')

nexttile
plot(resSumStats.Kdens(1).grid, resSumStats.Kdens(1).fk,'LineWidth',1.15)
title('Crude Oil')

nexttile
plot(resSumStats.Kdens(2).grid, resSumStats.Kdens(2).fk,'LineWidth',1.15)
title('Brent Oil')

nexttile
plot(resSumStats.Kdens(3).grid, resSumStats.Kdens(3).fk,'LineWidth',1.15)
title('Gasoline')

nexttile
plot(resSumStats.Kdens(4).grid, resSumStats.Kdens(4).fk,'LineWidth',1.15)
title('Heating Oil')


nexttile
plot(resSumStats.Kdens(5).grid, resSumStats.Kdens(5).fk,'LineWidth',1.15)
title('Gold')

nexttile
plot(resSumStats.Kdens(6).grid, resSumStats.Kdens(6).fk,'LineWidth',1.15)
title('Silver')

nexttile
plot(resSumStats.Kdens(7).grid, resSumStats.Kdens(7).fk,'LineWidth',1.15)
title('Platinum')

nexttile
plot(resSumStats.Kdens(8).grid, resSumStats.Kdens(8).fk,'LineWidth',1.15)
title('Palladium')


nexttile
plot(resSumStats.Kdens(9).grid, resSumStats.Kdens(9).fk,'LineWidth',1.15)
title('Copper')

nexttile
plot(resSumStats.Kdens(10).grid, resSumStats.Kdens(10).fk,'LineWidth',1.15)
title('Aluminium')

nexttile
plot(resSumStats.Kdens(11).grid, resSumStats.Kdens(11).fk,'LineWidth',1.15)
title('Lead')

nexttile
plot(resSumStats.Kdens(12).grid, resSumStats.Kdens(12).fk,'LineWidth',1.15)
title('Zinc')


nexttile
plot(resSumStats.Kdens(13).grid, resSumStats.Kdens(13).fk,'LineWidth',1.15)
title('Corn')

nexttile
plot(resSumStats.Kdens(14).grid, resSumStats.Kdens(14).fk,'LineWidth',1.15)
title('Soybean')

nexttile
plot(resSumStats.Kdens(15).grid, resSumStats.Kdens(15).fk,'LineWidth',1.15)
title('Soybean Oil')

nexttile
plot(resSumStats.Kdens(16).grid, resSumStats.Kdens(16).fk,'LineWidth',1.15)
title('Oats')

% Keep clean the workspace
clearvars tKdens

%% Heat maps

%tHmap = tiledlayout(1,2,'TileSpacing','compact');

t1 = tiledlayout(2,2,'TileSpacing','compact');
t2 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t3 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t4 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t5 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
% t6 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
% t7 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');

t3.Layout.Tile = 2;
t4.Layout.Tile = 3;
t5.Layout.Tile = 4;
% t6.Layout.Tile = 5;
% t7.Layout.Tile = 6;

xvals = {'T1','B1','RB1','HO1','GC1','SI1','PL1','PA1','HG1','AL1','PB1','ZN1','C1','S1','BO1','O1'};
yvals = {'T1','B1','RB1','HO1','GC1','SI1','PL1','PA1','HG1','AL1','PB1','ZN1','C1','S1','BO1','O1'};

corr_ret_preC = corrcoef(dataReturns.tmpRet(1:734,:));
corr_vol_preC = corrcoef(dataVol.tmpVol(1:734,:));
corr_ret_postC = corrcoef(dataReturns.tmpRet(735:end,:));
corr_vol_postC = corrcoef(dataVol.tmpVol(735:end,:));
% corr_ret_postRA = corrcoef(dataReturns.tmpRet(1295:end,:));
% corr_vol_postRA = corrcoef(dataVol.tmpVol(1295:end,:));

nexttile(t2)
heatmap(xvals,yvals,corr_ret_preC,'Colormap',jet,'GridVisible','off','CellLabelColor','none','ColorbarVisible','off',FontSize=17)
title(t2,'Returns correlations (pre)','FontWeight','bold','FontSize',20)

nexttile(t3)
heatmap(xvals,yvals,corr_vol_preC,'Colormap',jet,'GridVisible','off','CellLabelColor','none','ColorbarVisible','on',FontSize=17)
title(t3,'Volatilities correlations (pre)','FontWeight','bold','FontSize',20)

nexttile(t4)
heatmap(xvals,yvals,corr_ret_postC,'Colormap',jet,'GridVisible','off','CellLabelColor','none','ColorbarVisible','off',FontSize=17)
title(t4,'Returns correlations (post)','FontWeight','bold','FontSize',20)

nexttile(t5)
heatmap(xvals,yvals,corr_vol_postC,'Colormap',jet,'GridVisible','off','CellLabelColor','none','ColorbarVisible','on',FontSize=17)
title(t5,'Volatilities correlations (post)','FontWeight','bold','FontSize',20)

% nexttile(t6)
% heatmap(xvals,yvals,corr_ret_postRA,'Colormap',jet,'GridVisible','off','CellLabelColor','none','ColorbarVisible','off',FontSize=15)
% %title(t6,'Returns correlations (post)','FontWeight','bold','FontSize',20)
% 
% nexttile(t7)
% heatmap(xvals,yvals,corr_vol_postRA,'Colormap',jet,'GridVisible','off','CellLabelColor','none','ColorbarVisible','on',FontSize=15)
% %title(t7,'Volatilities correlations (post)','FontWeight','bold','FontSize',20)

% Keep clean the workspace
clearvars xvals yvals corr_ret_preC corr_vol_preC corr_ret_postC corr_vol_postC t1 t2 t3 t4 t5

%% IRFs

xgrid = 1:1:h;
t1 = tiledlayout(4,4,'TileSpacing','tight');
title(t1,'Generalized Impulse Responses','FontWeight','bold','FontSize',20)
xlabel(t1,'h')
%ylabel(tIRFs,'')

t2 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t3 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t4 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t5 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t6 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t7 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t8 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t9 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t10 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t11 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t12 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t13 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t14 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t15 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t16 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');
t17 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','tight');

t3.Layout.Tile = 2;
t4.Layout.Tile = 3;
t5.Layout.Tile = 4;
t6.Layout.Tile = 5;
t7.Layout.Tile = 6;
t8.Layout.Tile = 7;
t9.Layout.Tile = 8;
t10.Layout.Tile = 9;
t11.Layout.Tile = 10;
t12.Layout.Tile = 11;
t13.Layout.Tile = 12;
t14.Layout.Tile = 13;
t15.Layout.Tile = 14;
t16.Layout.Tile = 15;
t17.Layout.Tile = 16;

nexttile(t2)
plot(xgrid, VAR.IRFprint(1,:))
title('T1 -> T1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(1,:)) max(VAR.IRFprint(1,:))])

nexttile(t6)
plot(xgrid, VAR.IRFprint(2,:))
title('T1 -> B1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(2,:)) max(VAR.IRFprint(2,:))])

nexttile(t10)
plot(xgrid, VAR.IRFprint(3,:))
title('T1 -> GC1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(3,:)) max(VAR.IRFprint(3,:))])

nexttile(t14)
plot(xgrid, VAR.IRFprint(4,:))
title('T1 -> SI1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(4,:)) max(VAR.IRFprint(4,:))])


nexttile(t3)
plot(xgrid, VAR.IRFprint(5,:))
title('B1 -> T1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(5,:)) max(VAR.IRFprint(5,:))])

nexttile(t7)
plot(xgrid, VAR.IRFprint(6,:))
title('B1 -> B1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(6,:)) max(VAR.IRFprint(6,:))])

nexttile(t11)
plot(xgrid, VAR.IRFprint(7,:))
title('B1 -> GC1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(7,:)) max(VAR.IRFprint(7,:))])

nexttile(t15)
plot(xgrid, VAR.IRFprint(8,:))
title('B1 -> SI1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(8,:)) max(VAR.IRFprint(8,:))])



nexttile(t4)
plot(xgrid, VAR.IRFprint(9,:))
title('GC1 -> T1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(9,:)) max(VAR.IRFprint(9,:))])

nexttile(t8)
plot(xgrid, VAR.IRFprint(10,:))
title('GC1 -> B1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(10,:)) max(VAR.IRFprint(10,:))])

nexttile(t12)
plot(xgrid, VAR.IRFprint(11,:))
title('GC1 -> GC1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(11,:)) max(VAR.IRFprint(11,:))])

nexttile(t16)
plot(xgrid, VAR.IRFprint(12,:))
title('GC1 -> SI1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(12,:)) max(VAR.IRFprint(12,:))])



nexttile(t5)
plot(xgrid, VAR.IRFprint(13,:))
title('SI1 -> T1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(13,:)) max(VAR.IRFprint(13,:))])

nexttile(t9)
plot(xgrid, VAR.IRFprint(14,:))
title('SI1 -> B1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(14,:)) max(VAR.IRFprint(14,:))])


nexttile(t13)
plot(xgrid, VAR.IRFprint(15,:))
title('SI1 -> GC1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(15,:)) max(VAR.IRFprint(15,:))])

nexttile(t17)
plot(xgrid, VAR.IRFprint(16,:))
title('SI1 -> SI1','FontSize',14,'FontWeight','bold')
xlim([1 h]);
ylim([min(VAR.IRFprint(16,:)) max(VAR.IRFprint(16,:))])


% Keep clean the workspace
clearvars t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12 t13 t14 t15 t16 t17 xgrid

%% GARCH Mean reverting

lrvGARCH = (sqrt(resGARCH.LRVGARCH)*100)*ones(1,hGARCH);
lrvGJR = (sqrt(resGARCH.LRVGJR)*100)*ones(1,hGARCH);
yGARCH = sqrt(resGARCH.fcastGARCH(1330+1:end,:))*100;
yGJR = sqrt(resGARCH.fcastGJR(1330+1:end,:))*100;

t1 = tiledlayout(1,2,'TileSpacing','compact');
t2 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t3 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
%title(t1,'Mean reverting of Crude Oil','FontWeight','bold','FontSize',20)
xlabel(t1,'h')
ylabel(t1,'$\sigma_t$','Interpreter','latex')

t3.Layout.Tile = 2;
t4.Layout.Tile = 3;

nexttile(t2)
plot(lrvGARCH)
hold on
plot(yGARCH)
title('GARCH(1,1)','FontSize',10,'FontWeight','normal')
legend('$\sqrt{V_L}$','$\sigma_t$','Interpreter','latex','fontsize',11)
 ylim([3.5 4.1])
xlim([0 hGARCH])

nexttile(t3)
plot(lrvGJR)
hold on
plot(yGJR)
title('GJR-GARCH(1,1)','FontSize',10,'FontWeight','normal')
legend('$\sqrt{V_L}$','$\sigma_t$','Interpreter','latex','fontsize',11)
 ylim([2.5 4.5])
xlim([0 hGARCH])

% Keep clean the workspace
clear t1 t2 t3  yGARCH yGJR lrvGARCH lrvGJR
%% Heatmaps

xvals = [vars.namesVars(1:2) vars.namesVars(5:6) 'Total'];
yvals = [vars.namesVars(1:2) vars.namesVars(5:6) 'Total'];

% xvals = [vars.namesVars(1:8)  'Total'];
% yvals = [vars.namesVars(1:8)  'Total'];


% Spillovers btw T1, B1, GC1, SI1
t1 = tiledlayout(2,2,'TileSpacing','compact');
t2 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t3 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t4 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t5 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');

t3.Layout.Tile = 2;
t4.Layout.Tile = 3;
t5.Layout.Tile = 4;
t6.Layout.Tile = 5;

nexttile(t2)
heatmap(xvals,yvals,VAR.Spillovers(:,:,2),'Colormap',parula,'GridVisible','off', 'ColorbarVisible','off');
title(t2,'Spillovers h = 1','FontWeight','bold','FontSize',20)

nexttile(t3)
heatmap(xvals,yvals,VAR.Spillovers(:,:,6),'Colormap',parula,'GridVisible','off', 'ColorbarVisible','off');
title(t3,'Spillovers h = 5','FontWeight','bold','FontSize',20)

nexttile(t4)
heatmap(xvals,yvals,VAR.Spillovers(:,:,11),'Colormap',parula,'GridVisible','off', 'ColorbarVisible','off');
title(t4,'Spillovers h = 10','FontWeight','bold','FontSize',20)

nexttile(t5)
heatmap(xvals,yvals,VAR.Spillovers(:,:,23),'Colormap',parula,'GridVisible','off', 'ColorbarVisible','off');
title(t5,'Spillovers h = 22','FontWeight','bold','FontSize',20)


% Keep clean the workspace
clear t1 t2 t3 t4 t5 xvals yvals
%% Spillovers full K
xvals = [vars.namesVars(1:K) 'Total'];
yvals = [vars.namesVars(1:K) 'Total'];
heatmap(xvals,yvals,VAR.Spillovers(:,:,23),'Colormap',parula,'GridVisible','off', 'ColorbarVisible','off');

%% Scatter

t1 = tiledlayout(3,5,'TileSpacing','compact');
t2 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t3 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t4 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t5 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t6 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t7 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t8 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t9 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t10 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t11 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t12 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t13 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t14 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t15 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');
t16 = tiledlayout(t1,'flow','TileSpacing','tight','Padding','compact');



t3.Layout.Tile = 2;
t4.Layout.Tile = 3;
t5.Layout.Tile = 4;
t6.Layout.Tile = 5;
t7.Layout.Tile = 6;
t8.Layout.Tile = 7;
t9.Layout.Tile = 8;
t10.Layout.Tile = 9;
t11.Layout.Tile = 10;
t12.Layout.Tile = 11;
t13.Layout.Tile = 12;
t14.Layout.Tile = 13;
t15.Layout.Tile = 14;
t16.Layout.Tile = 15;
t17.Layout.Tile = 16;


nexttile(t2)
scatter(resIS_mlnHAR_CL1.h1_LHAR_PCA_levPCA1Energy.y,resIS_mlnHAR_CL1.h1_LHAR_PCA_levPCA1Energy.yhat,'filled')
hold on
scatter(resIS_NNHAR.h1_HARlevPCA1Energy.y,resIS_NNHAR.h1_HARlevPCA1Energy.yhat,'filled','MarkerFaceAlpha',0.3)
ylabel('h = 1')
title('Energy')


nexttile(t3)
scatter(resIS_mlnHAR_CL1.h1_LHAR_PCA_levPCA1PMetals.y,resIS_mlnHAR_CL1.h1_LHAR_PCA_levPCA1PMetals.yhat,'filled')
hold on
scatter(resIS_NNHAR.h1_HARlevPCA1PMetals.y,resIS_NNHAR.h1_HARlevPCA1PMetals.yhat,'filled','MarkerFaceAlpha',0.3)
title('Precious Metals')

nexttile(t4)
scatter(resIS_mlnHAR_CL1.h1_LHAR_PCA_levPCA1NFMetals.y,resIS_mlnHAR_CL1.h1_LHAR_PCA_levPCA1NFMetals.yhat,'filled')
hold on
scatter(resIS_NNHAR.h1_HARlevPCA1NFMetals.y,resIS_NNHAR.h1_HARlevPCA1NFMetals.yhat,'filled','MarkerFaceAlpha',0.3)
title('Non-ferrous Metals')


nexttile(t5)
scatter(resIS_mlnHAR_CL1.h1_LHAR_PCA_levPCA1Agric.y,resIS_mlnHAR_CL1.h1_LHAR_PCA_levPCA1Agric.yhat,'filled')
hold on
scatter(resIS_NNHAR.h1_HARlevPCA1Agric.y,resIS_NNHAR.h1_HARlevPCA1Agric.yhat,'filled','MarkerFaceAlpha',0.3)
title('Agriculture')


nexttile(t6)
scatter(resIS_mlnHAR_CL1.h1_LHAR_PCA_levPCA1Total.y,resIS_mlnHAR_CL1.h1_LHAR_PCA_levPCA1Total.yhat,'filled')
hold on
scatter(resIS_NNHAR.h1_HARlevPCA1Total.y,resIS_NNHAR.h1_HARlevPCA1Total.yhat,'filled','MarkerFaceAlpha',0.3)
title('Total')
legend('PCA','NN-PCA')


nexttile(t7)
scatter(resIS_mlnHAR_CL1.h5_LHAR_PCA_levPCA1Energy.y,resIS_mlnHAR_CL1.h5_LHAR_PCA_levPCA1Energy.yhat,'filled')
hold on
scatter(resIS_NNHAR.h5_HARlevPCA1Energy.y,resIS_NNHAR.h5_HARlevPCA1Energy.yhat,'filled','MarkerFaceAlpha',0.3)
ylabel('h = 5')

nexttile(t8)
scatter(resIS_mlnHAR_CL1.h5_LHAR_PCA_levPCA1PMetals.y,resIS_mlnHAR_CL1.h5_LHAR_PCA_levPCA1PMetals.yhat,'filled')
hold on
scatter(resIS_NNHAR.h5_HARlevPCA1PMetals.y,resIS_NNHAR.h5_HARlevPCA1PMetals.yhat,'filled','MarkerFaceAlpha',0.3)

nexttile(t9)
scatter(resIS_mlnHAR_CL1.h5_LHAR_PCA_levPCA1NFMetals.y,resIS_mlnHAR_CL1.h5_LHAR_PCA_levPCA1NFMetals.yhat,'filled')
hold on
scatter(resIS_NNHAR.h5_HARlevPCA1NFMetals.y,resIS_NNHAR.h5_HARlevPCA1NFMetals.yhat,'filled','MarkerFaceAlpha',0.3)

nexttile(t10)
scatter(resIS_mlnHAR_CL1.h5_LHAR_PCA_levPCA1Agric.y,resIS_mlnHAR_CL1.h5_LHAR_PCA_levPCA1Agric.yhat,'filled')
hold on
scatter(resIS_NNHAR.h5_HARlevPCA1Agric.y,resIS_NNHAR.h5_HARlevPCA1Agric.yhat,'filled','MarkerFaceAlpha',0.3)

nexttile(t11)
scatter(resIS_mlnHAR_CL1.h5_LHAR_PCA_levPCA1Total.y,resIS_mlnHAR_CL1.h5_LHAR_PCA_levPCA1Total.yhat,'filled')
hold on
scatter(resIS_NNHAR.h5_HARlevPCA1Total.y,resIS_NNHAR.h5_HARlevPCA1Total.yhat,'filled','MarkerFaceAlpha',0.3)
legend('PCA','NN-PCA')



nexttile(t12)
scatter(resIS_mlnHAR_CL1.h22_LHAR_PCA_levPCA1Energy.y,resIS_mlnHAR_CL1.h22_LHAR_PCA_levPCA1Energy.yhat,'filled')
hold on
scatter(resIS_NNHAR.h22_HARlevPCA1Energy.y,resIS_NNHAR.h22_HARlevPCA1Energy.yhat,'filled','MarkerFaceAlpha',0.3)
ylabel('h = 22')

nexttile(t13)
scatter(resIS_mlnHAR_CL1.h22_LHAR_PCA_levPCA1PMetals.y,resIS_mlnHAR_CL1.h22_LHAR_PCA_levPCA1PMetals.yhat,'filled')
hold on
scatter(resIS_NNHAR.h22_HARlevPCA1PMetals.y,resIS_NNHAR.h22_HARlevPCA1PMetals.yhat,'filled','MarkerFaceAlpha',0.3)

nexttile(t14)
scatter(resIS_mlnHAR_CL1.h22_LHAR_PCA_levPCA1NFMetals.y,resIS_mlnHAR_CL1.h22_LHAR_PCA_levPCA1NFMetals.yhat,'filled')
hold on
scatter(resIS_NNHAR.h22_HARlevPCA1NFMetals.y,resIS_NNHAR.h22_HARlevPCA1NFMetals.yhat,'filled','MarkerFaceAlpha',0.3)

nexttile(t15)
scatter(resIS_mlnHAR_CL1.h22_LHAR_PCA_levPCA1Agric.y,resIS_mlnHAR_CL1.h22_LHAR_PCA_levPCA1Agric.yhat,'filled')
hold on
scatter(resIS_NNHAR.h22_HARlevPCA1Agric.y,resIS_NNHAR.h22_HARlevPCA1Agric.yhat,'filled','MarkerFaceAlpha',0.3)


nexttile(t16)
scatter(resIS_mlnHAR_CL1.h22_LHAR_PCA_levPCA1Total.y,resIS_mlnHAR_CL1.h22_LHAR_PCA_levPCA1Total.yhat,'filled')
hold on
scatter(resIS_NNHAR.h22_HARlevPCA1Total.y,resIS_NNHAR.h22_HARlevPCA1Total.yhat,'filled','MarkerFaceAlpha',0.3)
legend('PCA','NN-PCA')
