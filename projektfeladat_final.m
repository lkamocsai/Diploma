% Init workspace
clear all;
clearvars;
clc;

% Set forecast horizons
vars.fhorizon = [1 5 22];

% Set window size for OOS forecast
vars.winSize = 630;

% Set Model names
vars.namesBZ = {'h1_BZ';'h5_BZ';'h22_BZ'};
vars.namesHAR = {'h1_HAR';'h5_HAR';'h22_HAR'};
vars.namesLHAR = {'h1_LHAR';'h5_LHAR';'h22_LHAR'};
vars.namesLHARPCA = {'h1_LHAR_PCA_';'h5_LHAR_PCA_';'h22_LHAR_PCA_'};

% Set variable names
vars.namesVars = {'T1','B1','RB1','HO1','GC1','SI1','PL1','PA1','HG1','AL1','PB1','ZN1','C1','S1','BO1','O1'};
%%
% Load data
%
% % ENERGY
% dataRaw.ICE_T1 =  tsTTimport('ICE_T1_FW.xlsx'); %ICE Crude front-month futures 
% dataRaw.ICE_B1 =  tsTTimport('ICE_B1_FW.xlsx'); %ICE Brent front-month futures
% dataRaw.CME_RB1 = tsTTimport('CME_RB1_FW.xlsx'); %NYMEX Gasoline front-month futures
% dataRaw.CME_HO1 = tsTTimport('CME_HO1_FW.xlsx'); %NYMEX Heating Oil front-month futures
% %dataRaw.CME_NG1 = tsTTimport('CME_NG1_FW.xlsx'); %NYMEX Natural Gas front-month futures
% 
% % METALS
% dataRaw.CME_GC1 = tsTTimport('CME_GC1_FW.xlsx'); %NYMEX Gold front-month futures
% dataRaw.CME_SI1 = tsTTimport('CME_SI1_FW.xlsx'); %NYMEX Silver front-month futures
% dataRaw.CME_PL1 = tsTTimport('CME_PL1_FW.xlsx'); %NYMEX Platinum front-month futures
% dataRaw.CME_PA1 = tsTTimport('CME_PA1_FW.xlsx'); %NYMEX Palladium front-month futures
% 
% % Non-ferious Metals
% dataRaw.CME_HG1 = tsTTimport('CME_HG1_FW.xlsx'); %COMEX Copper front-month futures
% dataRaw.SHFE_AL1 = tsTTimport('SHFE_AL1_FW.xlsx'); %SHFE Aluminium front-month futures
% dataRaw.SHFE_PB1 = tsTTimport('SHFE_PB1_FW.xlsx'); %SHFE Lead front-month futures
% dataRaw.SHFE_ZN1 = tsTTimport('SHFE_ZN1_FW.xlsx'); %SHFE Zinc front-month futures
% 
% % GRAINS
% dataRaw.CME_C1 = tsTTimport('CME_C1_FW.xlsx'); %CBOT Corn front-month futures
% dataRaw.CME_S1 = tsTTimport('CME_S1_FW.xlsx'); %CBOT Soybeans front-month futures
% dataRaw.CME_BO1 = tsTTimport('CME_BO1_FW.xlsx'); %CBOT Soybean Oil front-month futures
% dataRaw.CME_O1 = tsTTimport('CME_O1_FW.xlsx'); %CBOT Oats front-month futures

load projektfeladatData.mat

%%
% Clean data
vars.rawDataNames = fieldnames(dataRaw);
for i = 1:length(vars.rawDataNames)
    dataCleaned.([vars.rawDataNames{i} '_c']) = tspreprocdata(dataRaw.(vars.rawDataNames{i})(:,1:4));
end

% Sync data
vars.cleanedDataNames = fieldnames(dataCleaned);
for i = 1:length(vars.cleanedDataNames)
    dataCleaned.(vars.cleanedDataNames{i}) = synchronize(dataCleaned.(vars.cleanedDataNames{i}),dataCleaned.CME_GC1_c,"last","linear");
    dataCleaned.(vars.cleanedDataNames{i}) = dataCleaned.(vars.cleanedDataNames{i})(:,1:4);
    dataCleaned.(vars.cleanedDataNames{i}).Properties.VariableNames(1:4) = {'OPEN','HIGH','LOW','SETTLE'};
end
%%
% Create date vector
vars.dtp = dataCleaned.CME_GC1_c.DATE;

% Create price vectors and matrix
for i = 1:length(vars.cleanedDataNames)
    dataPrices.(['Pt_' vars.rawDataNames{i}]) = dataCleaned.(vars.cleanedDataNames{i}).SETTLE;
    dataStats.Pt(:,i) = dataCleaned.(vars.cleanedDataNames{i}).SETTLE;
end
%%
% Compute log returns
vars.dtr = dataCleaned.CME_GC1_c.DATE(2:end);

vars.pricesDataNames = fieldnames(dataPrices);
for i = 1:length(vars.pricesDataNames)
    dataReturns.(['rt_' vars.rawDataNames{i}]) = diff(log(dataCleaned.(vars.cleanedDataNames{i}).SETTLE))*100;
    dataStats.rt(:,i) = diff(log(dataStats.Pt(:,i)))*100;
end

vars.returnDataNames = fieldnames(dataReturns);
for i = 1:length(vars.returnDataNames)
    dataReturns.tmpRet(:,i) = dataReturns.(vars.returnDataNames{i});
end

% Compute squared returns (r(t)^2 = (y(t) - y(t)bar)^2)
for i = 1:size(dataStats.Pt,2)
    dataStats.rt2(:,i) = dataStats.rt(:,i) - mean(dataStats.rt(:,i));
    dataStats.rt2(:,i) = dataStats.rt2(:,i).^2;
end
%%
% Compute volatility series
vars.annmult = 1*(100^2); %annual multiplier
vars.volMeasureSel = 3;
for i = 1:length(vars.cleanedDataNames)
    dataVol.(['RVd_' vars.rawDataNames{i}]) = [sqrt(vars.annmult*(tsRangeVolEst(dataCleaned.(vars.cleanedDataNames{i})(1:end,:),'SQ')))...
                                               sqrt(vars.annmult*(tsRangeVolEst(dataCleaned.(vars.cleanedDataNames{i})(1:end,:),'PKJ')))...
                                               sqrt(vars.annmult*(tsRangeVolEst(dataCleaned.(vars.cleanedDataNames{i})(2:end,:),'GK')))...
                                               sqrt(vars.annmult*(tsRangeVolEst(dataCleaned.(vars.cleanedDataNames{i})(2:end,:),'RS')))...
                                               sqrt(vars.annmult*(tsRangeVolEst(dataCleaned.(vars.cleanedDataNames{i})(1:end,:),'MXV')))];
end

% Get field names
vars.volDataNames = fieldnames(dataVol);

% Create volatility matrix
for i = 1:length(vars.volDataNames)
    dataVol.tmpVol(:,i) = dataVol.(vars.volDataNames{i})(:,vars.volMeasureSel);
end
%%
% Compute correlation matrices
tmpdataLevCov.covLevEnergy =   cov( dataReturns.tmpRet(:,2:4));
tmpdataLevCov.covLevPMetals =  cov( dataReturns.tmpRet(:,5:8));
tmpdataLevCov.covLevNFMetals = cov( dataReturns.tmpRet(:,9:12));
tmpdataLevCov.covLevAgric =    cov( dataReturns.tmpRet(:,13:end));
tmpdataLevCov.covLevTotal =    cov( dataReturns.tmpRet(:,2:end));

[~,dataLevPCAs.levPCA1Energy,~,~,~,~] = pca( dataReturns.tmpRet(:,2:4));
[~,dataLevPCAs.levPCA1PMetals,~,~,~,~] = pca( dataReturns.tmpRet(:,5:8) );
[~,dataLevPCAs.levPCA1NFMetals,~,~,~,~] = pca( dataReturns.tmpRet(:,9:12) );
[~,dataLevPCAs.levPCA1Agric,~,~,~,~] = pca( dataReturns.tmpRet(:,13:end) );
[~,dataLevPCAs.levPCA1Total,~,~,~,~] = pca( dataReturns.tmpRet(:,2:end) );

dataLevPCAs.levPCA1Energy = dataLevPCAs.levPCA1Energy(:,1);
dataLevPCAs.levPCA1PMetals = dataLevPCAs.levPCA1PMetals(:,1);
dataLevPCAs.levPCA1NFMetals = dataLevPCAs.levPCA1NFMetals(:,1);
dataLevPCAs.levPCA1Agric = dataLevPCAs.levPCA1Agric(:,1);
dataLevPCAs.levPCA1Total = dataLevPCAs.levPCA1Total(:,1);

% Compute eigenvectors, eigenvalues, and the explained variation
vars.levCovDataNames = fieldnames(tmpdataLevCov);
for i = 1:length(vars.levCovDataNames)
    [tmpestimLevPCA.(['eigvec_'     (vars.levCovDataNames{i})]),...
     tmpestimLevPCA.(['eigval_'     (vars.levCovDataNames{i})]),...
     tmpestimLevPCA.(['explained_'  (vars.levCovDataNames{i})])] = pcacov(tmpdataLevCov.(vars.levCovDataNames{i}));
end

% Keep clean the workspace
clearvars tmpdataLevCov tmpestimLevPCA
%%
% Creating HAR type volatility datasets
lowB = 0;
highB = 1;
for i = 1:length(vars.volDataNames)
    tmplogRV = log(dataVol.(vars.volDataNames{i})(:,vars.volMeasureSel));
    tmplogRV = tsRNormData(tmplogRV,lowB,highB);
    dataHARRV.(['X_' vars.volDataNames{i}]) = tsCreateHARdataset(tmplogRV);
end

% Keep clean the workspace
clearvars lowB highB tmplogRV i
%%
% Create HAR return dataset to use as an input for leverage series
for i = 1:length(vars.returnDataNames)
    dataHARReturns.(['X_' vars.returnDataNames{i}]) = tsCreateHARdataset(dataReturns.(vars.returnDataNames{i}));
end

% Creating HAR leverage dataset
lowB = -1;
highB = 0;
vars.returnHARDataNames = fieldnames(dataHARReturns);
for i = 1:length(vars.returnDataNames)
    tmpHARLev = dataHARReturns.(vars.returnHARDataNames{i}).*(dataHARReturns.(vars.returnHARDataNames{i})<0);
    dataHARLev.(['X_lev_' vars.returnDataNames{i}]) = tsRNormData(tmpHARLev,lowB,highB);
end

% Keep clean the workspace
clearvars lowB highB tmpHARLev i
%%
% Creating leverage PCA HAR dataset
lowB = -1;
highB = 0;
vars.levHARDataNames = fieldnames(dataLevPCAs);
for i = 1:length(vars.levHARDataNames)
    tmpHARLevPCA = tsCreateHARdataset(dataLevPCAs.(vars.levHARDataNames{i}));
    tmpHARLevPCA = tmpHARLevPCA.*(tmpHARLevPCA < 0);
    dataHARLevPCAs.(['X_' vars.levHARDataNames{i}]) = tsRNormData(tmpHARLevPCA,lowB,highB);
end

% Keep clean the workspace
clearvars lowB highB tmpHARLevPCA i
%%
% Run descriptive statistics

% Returns statistics
resSumStats.stats_rt = tsbasetats(dataStats.rt);
resSumStats.sumStats_rt = tsSummaryStats(dataStats.rt);

% Square returns statistics
resSumStats.stats_rt2 = tsbasetats(dataStats.rt2);

% Volatility statistics
resSumStats.stats_RV = tsbasetats( sqrt(dataVol.tmpVol));
resSumStats.sumStats_RV = tsSummaryStats(sqrt(dataVol.tmpVol));

% Pre vol stat
resSumStats.stats_pre_RV = tsbasetats( sqrt(dataVol.tmpVol(1:734,:) ));
resSumStats.sumStats_pre_RV = tsSummaryStats(sqrt(dataVol.tmpVol(1:734,:) ));

% Post vol stat
resSumStats.stats_post_RV = tsbasetats( sqrt(dataVol.tmpVol(735:end,:) ));
resSumStats.sumStats_post_RV = tsSummaryStats(sqrt(dataVol.tmpVol(735:end,:) ));

% Roll skew
for i = 1:size(dataVol.tmpVol,2)
    BBBB(:,i,1) = tsRollSkew( sqrt(dataVol.tmpVol(:,i)),252);
    BBBB(:,i,2) = tsRollKurt( sqrt(dataVol.tmpVol(:,i)),252);
end

% Calc ACFs for rt, rt^2 and RV
for i = 1:size(dataStats.rt,2)
    resSumStats.ACF_rt(:,i) = autocorr(dataStats.rt(:,i),NumLags=66);
    resSumStats.ACF_rt2(:,i) = tsACF(dataStats.rt2(:,i),66);
    resSumStats.ACF_RV(:,i) = tsACF(sqrt(252*dataVol.tmpVol(:,i)),66); %tsACF(dataVol.tmpVol(:,i),66);
end

% Estimate densities 
K = size(dataStats.rt,2);
for i = 1:K
    resSumStats.Kdens(i).min_ts = min(dataStats.rt(:,i));
    resSumStats.Kdens(i).max_ts = max(dataStats.rt(:,i));
    resSumStats.Kdens(i).grid = (resSumStats.Kdens(i).min_ts:0.001:resSumStats.Kdens(i).max_ts)';
    resSumStats.Kdens(i).fk = tsNonParEst(dataStats.rt(:,i),resSumStats.Kdens(i).grid);
end
%%
% In-sample estimates

% Set data
dsetHARmain =   dataHARRV.X_RVd_ICE_T1; % main HAR data
dsetHARlevrt =  dataHARLev.X_lev_rt_ICE_T1; % leverage HAR data

% Joint Wald test restriction
df = 3;
Q = [0;0;0];
R = [0 0 0 0 0 0 0 1 0 0 0;
     0 0 0 0 0 0 0 0 1 0 0;
     0 0 0 0 0 0 0 0 0 1 0];

% Start estimate
for j = 1: length(vars.fhorizon)
   % Benchmark 'Original' HAR model
   tmpDset = (dsetHARmain);
   resIS_mlnHAR_CL1.([vars.namesHAR{j}]) = tsMLEstimateHAR(tmpDset,vars.fhorizon(j),'normal',0.1);

   % 'Original' Corsi-Reno leverage HAR model (LHAR)
   tmpDset = [(dsetHARmain) dsetHARlevrt];
   resIS_mlnHAR_CL1.([vars.namesLHAR{j}]) = tsMLEstimateHAR(tmpDset,vars.fhorizon(j),'normal',0.1);

   % PCA models per segment
    for i = 1:length(vars.levHARDataNames)
        % Leverage PCA HAR model (PCA-LHAR)
        tmpDset = [(dsetHARmain) dsetHARlevrt dataHARLevPCAs.(['X_' vars.levHARDataNames{i}])]; 
        resIS_mlnHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}])= tsMLEstimateHAR(tmpDset,vars.fhorizon(j),'normal',0.1);
        tmpName = [vars.namesLHARPCA{j}   vars.levHARDataNames{i}];
        [resIS_mlnHAR_CL1.(tmpName).WG.val,resIS_mlnHAR_CL1.(tmpName).WG.pv,resIS_mlnHAR_CL1.(tmpName).WG.Hip]=tsWaldtest(resIS_mlnHAR_CL1.(tmpName).theta1,...
                                                                                                                     size(resIS_mlnHAR_CL1.(tmpName).y,1),...
                                                                                                                     resIS_mlnHAR_CL1.(tmpName).Ht1,R,Q,df,0.05);
    end
end

% Keep clean the workspace
clearvars tmpName R Q df i j
%%
% Run Out-of-Sample forecast

% Start OOS
for j = 1: length(vars.namesHAR)
   % Benchmark 'Original' HAR model
   tmpDset = (dsetHARmain);
   resOOS_olsHAR_CL1.([vars.namesHAR{j}])   = tsHARforecastOLS(tmpDset,vars.fhorizon(j),vars.winSize);

   % 'Original' Corsi-Reno leverage HAR model (LHAR)
   tmpDset = [(dsetHARmain)  dsetHARlevrt(:,:)];
   resOOS_olsHAR_CL1.([vars.namesLHAR{j}])   = tsHARforecastOLS(tmpDset,vars.fhorizon(j),vars.winSize);

   % PCA models
    for i = 1:length(vars.levHARDataNames)
        % Leverage PCA HAR model (PCA-LHAR)
        tmpDset = [(dsetHARmain)  dsetHARlevrt(:,:) dataHARLevPCAs.(['X_' vars.levHARDataNames{i}])];
        resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}])= tsHARforecastOLS(tmpDset,vars.fhorizon(j),vars.winSize);
    end
end
%%
% GARCH(1,1) and GJR-GARCH(1,1) models

% Load price series
yt = dataPrices.Pt_ICE_T1;

% Init parameters vector
theta0 = [0.001 0.01 0.1 0.6 0.2];

% Estimate GARCH models
[resGARCH.thetaG,~,resGARCH.tstatG,resGARCH.fitG,resGARCH.MSEG] = tsMLEstimateGARCH(yt,theta0(1:end-1),'GARCH');
[resGARCH.thetaGJR,~,resGARCH.tstatGJR,resGARCH.fitGJR,resGARCH.MSEGJR] = tsMLEstimateGARCH(yt,theta0,'GJR');

% Forecast GARCH
hGARCH = 252;
[resGARCH.fcastGARCH, resGARCH.LRVGARCH] = tsForecastGARCH(resGARCH.fitG,resGARCH.thetaG,hGARCH,"GARCH");
[resGARCH.fcastGJR,resGARCH.LRVGJR] = tsForecastGARCH(resGARCH.fitGJR,resGARCH.thetaGJR,hGARCH,"GJR");

% Save results
table3 = nan(4,5);
table3(1,1:end-1) = resGARCH.thetaG;
table3(2,1:end-1) = resGARCH.tstatG;
table3(3,1:end) = resGARCH.thetaGJR;
table3(4,1:end) = resGARCH.tstatGJR;

% Keep clean the workspace
clear theta0 yt hGARCH
%% 
% BZ model

%Bollerslev-Zhou model

% Get returns and leverage
y = diff(log(dataPrices.Pt_ICE_T1))*100;
yl = (y).*(y<0);

% Normalize data
yl = tsRNormData(yl,-1,0);
y = tsRNormData(y,-1,1);
vData = tsRNormData( log(dataVol.RVd_ICE_T1(:,3)),0,1);

% In-Sample estimate 
for i = 1: length(vars.fhorizon)
   resIS_mlnBZ_T1.([vars.namesBZ{i}]) = tsBZfitML(vData,y,yl,vars.fhorizon(i),'normal',0.05);
end

% Out-of-Sample forecast
for i = 1: length(vars.fhorizon)
   resOOS_mlnBZ_T1.([vars.namesBZ{i}]) = tsBZforecastML(vData,y,yl,vars.fhorizon(i),vars.winSize);
end

% Keep clean the workspace
clearvars y yl vData i
%%
% Neural Network with jump connection
 
% Set data
dsetNNHARmain = dataHARRV.X_RVd_ICE_T1; 
dsetNNHARlevrt = dataHARLev.X_lev_rt_ICE_T1;

% Start estimates
for j = 1: length(vars.fhorizon)
    for i = 1:length(vars.levHARDataNames)

        % In-Sample estimates
        tmpDset = [dsetNNHARmain dsetNNHARlevrt dataHARLevPCAs.(['X_' vars.levHARDataNames{i}])];
        NNy = tmpDset(1 + vars.fhorizon(j):end,j);
        NNx = tmpDset(1:end - vars.fhorizon(j),:);
        XX = [ones(size(NNy,1),1) NNx tsDeepNeuralNet(NNx,NNy,0.1,[10;100;10],'tanh','n')']; 
        yy = NNy;
        T = size(yy,1);
        resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]) = nwest(yy(22:end,:),XX(22:end,:),2*vars.fhorizon(j));
        resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).pval = 2*(1-tcdf(abs( resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).tstat ),T-2));
        resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MSE = mean((resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).y -resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).yhat).^2) ;
        resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MZ = nwest(resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).y,[ones(size(resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).yhat,1),1) resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).yhat]);

        % Save the results
        table2a(1:2:21,i,j) = resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).beta;
        table2a(2:2:22,i,j) = resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).pval;
        table2a(23,i,j) = resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MSE;
        table2a(24,i,j) = resIS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MZ.beta(2);

        % Out-of-Sample estimates
        for w = vars.winSize + 1:T
            % Set rolling window
            tmpX = XX(w-vars.winSize:w-1,:);
            tmpy = yy(w-vars.winSize:w-1,:);
            X = tmpX(22:end,:);
            y = tmpy(22:end,:);
        
            % Estimate model
            bhatrw(w,:) = (X(1:end,:)'*X(1:end,:))\(X(1:end,:)'*y(1:end));
            yhatrw(w,1) = dsetNNHARmain(w,j);
            yhatrw(w,2) = X(end-vars.fhorizon(j)+1,:)*bhatrw(w,:)';
        end

        % Save results
        resOOS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).yhat = yhatrw(vars.winSize+1:end,:,:);
        resOOS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).bhat = bhatrw(vars.winSize+1:end,:,:);
        resOOS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MSEs = ((resOOS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).yhat(:,1)-resOOS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).yhat(:,2)).^2);
        resOOS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MSE = mean((resOOS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).yhat(:,1)-resOOS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).yhat(:,2)).^2);
        table4a(:,i,j) = resOOS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MSE;
        
        % Collect PCA-LHAR Losses
        resOOSNNLoss(:,i+3,j) = resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MSEs(1:end-1,1);
        % Collect NN-PCA-LHAR Losses
        resOOSNNLoss(:,i+8,j) = resOOS_NNHAR.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MSEs;
    end
end

% Insert base models
wintype = 1;
for j = 1: length(vars.namesHAR)
    resOOSNNLoss(:,1,j)   = resOOS_mlnBZ_T1.(vars.namesBZ{j}).MSEs(1:end-1,:);
    resOOSNNLoss(:,2,j)   = resOOS_olsHAR_CL1.(vars.namesHAR{j}).MSEs(1:end-1,wintype);
    resOOSNNLoss(:,3,j)   = resOOS_olsHAR_CL1.(vars.namesLHAR{j}).MSEs(1:end-1,wintype);
end

% Run MCS and save results
SPANN = nan(size(resOOSNNLoss,3),size(resOOSNNLoss,2));
for jj = 1:size(resOOSNNLoss,3)
    for ii = 1:size(resOOSNNLoss,2)
        benchNNLoss = resOOSNNLoss(:,ii,jj);
        modelsNNLoss = resOOSNNLoss(:,:,jj);
        modelsNNLoss(:,ii) =[];
        SPANN(jj,ii) = bsds(benchNNLoss,modelsNNLoss,10000,9,'BLOCK');
    end
end

% Keep clean the workspace 
clearvars dsetNNHARmain dsetNNHARlevrt NNy NNx XX yy yhatrw bhatrw tmpX tmpy X y modelsNNLoss benchNNLoss jj ii
%%
% Model Confidence Set

% Forecast window type
wintype = 1;

% Collect losses
resOOSLoss=nan(size(resOOS_olsHAR_CL1.h1_HAR.uhat,1),7+1,3);
for j = 1: length(vars.namesHAR)
    resOOSLoss(:,1,j)   = resOOS_mlnBZ_T1.(vars.namesBZ{j}).MSEs;
    resOOSLoss(:,2,j)   = resOOS_olsHAR_CL1.(vars.namesHAR{j}).MSEs(:,wintype);
    resOOSLoss(:,3,j)   = resOOS_olsHAR_CL1.(vars.namesLHAR{j}).MSEs(:,wintype);
    for i = 1:length(vars.levHARDataNames)
        resOOSLoss(:,i+3,j) =   resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MSEs(:,wintype);
    end  
end

% Run MCS and save results
for jj = 1:size(resOOSLoss,3)
    for ii = 1:size(resOOSLoss,2)
        benchLoss = resOOSLoss(:,ii,jj);
        modelsLoss = resOOSLoss(:,:,jj);
        modelsLoss(:,ii) =[];
        SPA(jj,ii) = bsds(benchLoss,modelsLoss,10000,9,'BLOCK');
    end
end

% Keep clean the workspace
clearvars wintype hh jj ii benchLoss modelsLoss
%%
% VAR model and Spillovers
y = ([dataVol.RVd_ICE_T1(:,3) dataVol.RVd_ICE_B1(:,3)  dataVol.RVd_CME_GC1(:,3) dataVol.RVd_CME_SI1(:,3) ]);
y = tsRNormData(y,0,1);
x = [dataHARLevPCAs.X_levPCA1Energy(:,1) dataHARLevPCAs.X_levPCA1PMetals(:,1) ];

vardata = [y x];
K = size(y,2);
M = size(x,2);
p = 2;
s = 0;
h = 25;

% Estimate VAR 
[VAR.mu,VAR.A,VAR.B,VAR.SIGMA,~,VAR.Z,VAR.tvals,VAR.pvals] = tsEstimateVARX(y,x,p,s);
% Generalized IRFs
[VAR.PHI,VAR.IRFGen,VAR.DynMult] = tsIRF2(VAR.A,VAR.SIGMA,p,h,"IRFGen","s",s,"M",M,"B",VAR.B);
% Generalized FEVD
VAR.FEVD = tsFEVD2(VAR.PHI,VAR.SIGMA,VAR.IRFGen,h);
% Spillovers
VAR.Spillovers = tsConnectedness(VAR.FEVD);

% Reshape IRFs to print them
for i = 1:h
    VAR.IRFprint(:,i) = reshape(VAR.IRFGen(:,:,i),K^2,1);
    VAR.DynMultprint(:,i) = reshape(   VAR.DynMult(:,:,i),K*size(x,2),1);
end

% Report VAR estimate results
table6 = nan(2*(K*p + M*(s + 1) + 1),K);
tmpCoeffs = [VAR.mu VAR.A(1:K,:) VAR.B(1:K,:)];
nCols = size(tmpCoeffs,2)*2;

for i = 1:K
    table6(1:2:nCols,i) = tmpCoeffs(i,:);
    table6(2:2:nCols,i) = VAR.pvals(i,:);
end

% Keep clean the workspace
clearvars tmpCoeffs nCols vardata K M p s
%%
% In-Sample estimates report

% Init table
table2=nan(10*2+4,7,3);

% Save results
for j = 1: length(vars.namesHAR)
    table2(1:2:7,1,j) = resIS_mlnHAR_CL1.(vars.namesHAR{j}).theta1(1:end-1);
    table2(2:2:8,1,j) = resIS_mlnHAR_CL1.(vars.namesHAR{j}).W.pv1; 
    table2(22,1,j) =    resIS_mlnHAR_CL1.(vars.namesHAR{j}).RMSE; 
    table2(23,1,j) =    resIS_mlnHAR_CL1.(vars.namesHAR{j}).MZ.beta(2); 
    table2(24,1,j) =    resIS_mlnHAR_CL1.(vars.namesHAR{j}).MZ.rbar; 

    table2(1:2:13,2,j) = resIS_mlnHAR_CL1.(vars.namesLHAR{j}).theta1(1:end-1);
    table2(2:2:14,2,j) = resIS_mlnHAR_CL1.(vars.namesLHAR{j}).W.pv1; 
    table2(22,2,j) =     resIS_mlnHAR_CL1.(vars.namesLHAR{j}).RMSE; 
    table2(23,2,j) =     resIS_mlnHAR_CL1.(vars.namesLHAR{j}).MZ.beta(2); 
    table2(24,2,j) =     1 - (resIS_mlnHAR_CL1.(vars.namesLHAR{j}).RMSE / resIS_mlnHAR_CL1.(vars.namesHAR{j}).RMSE);

    for i = 1:length(vars.levHARDataNames)
        table2(1:2:19,i+2,j) =   resIS_mlnHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).theta1(1:end-1);
        table2(2:2:20,i+2,j) =   resIS_mlnHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).W.pv1;
        table2(22,i+2,j) =       resIS_mlnHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).RMSE;
        table2(23,i+2,j) =       resIS_mlnHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MZ.beta(2); 
        table2(24,i+2,j) =       1 - (resIS_mlnHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).RMSE/ resIS_mlnHAR_CL1.(vars.namesHAR{j}).RMSE);
    end
end

% Print results
for i = 1:3
    clear info;
    info.cnames = char('HAR','LHAR','PCA-LHAR-En','PCA-LHAR-PMet','PCA-LHAR-NFMet','PCA-LHAR-Agric','PCA-Total');
    info.rnames = char('Table2','b0','','b1d','','b2w','','b3m','','blev1d','','blev2w','','blev3m','','bpca1d','','bpca2w','','bpca3m','','','MSE','MZ beta','MZ R2');
    info.fmt    = '%10.4f';
    sprintf('\n\n');
    mprint(table2(:,:,i),info);
end
%%
% Out-of-sample estimates report

% Window type = rolling
wt = 1; 

% Init table
table4 = nan(6,7,3);

% Save results
for j = 1: length(vars.fhorizon)
    table4(1:4,1,j) = [resOOS_olsHAR_CL1.(vars.namesHAR{j}).MSE(wt)...
                       resOOS_olsHAR_CL1.(vars.namesHAR{j}).HRMSE(wt)...
                       resOOS_olsHAR_CL1.(vars.namesHAR{j}).MZ(wt).beta(2)...
                       resOOS_olsHAR_CL1.(vars.namesHAR{j}).MZ(wt).rbar];

    table4(1:4,2,j) = [resOOS_olsHAR_CL1.(vars.namesLHAR{j}).MSE(wt)/resOOS_olsHAR_CL1.(vars.namesHAR{j}).MSE(wt)...
                       resOOS_olsHAR_CL1.(vars.namesLHAR{j}).HRMSE(wt)/resOOS_olsHAR_CL1.(vars.namesHAR{j}).HRMSE(wt)...
                       resOOS_olsHAR_CL1.(vars.namesLHAR{j}).MZ(wt).beta(2)...
                       1-(resOOS_olsHAR_CL1.(vars.namesLHAR{j}).MSE(wt)/resOOS_olsHAR_CL1.(vars.namesHAR{j}).MSE(wt)) ]; %resOOS_olsHAR_CL1.(vars.namesLHAR{j}).MZ(wt).rbar];
    [table4(5,2,j), table4(6,2,j)] = tsCWtest(resOOS_olsHAR_CL1.(vars.namesHAR{j}).uhat(:,wt),resOOS_olsHAR_CL1.(vars.namesLHAR{j}).uhat(:,wt),...
                                              resOOS_olsHAR_CL1.(vars.namesHAR{j}).yhat(:,2,wt), resOOS_olsHAR_CL1.(vars.namesLHAR{j}).yhat(:,2,wt), vars.fhorizon(j));

    for i = 1:length(vars.levHARDataNames)
        table4(1:4,i+2,j) = [resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MSE(wt)/resOOS_olsHAR_CL1.(vars.namesHAR{j}).MSE(wt)...
                             resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).HRMSE(wt)/resOOS_olsHAR_CL1.(vars.namesHAR{j}).HRMSE(wt)...
                             resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MZ(wt).beta(2)...
                             1-(resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MSE(wt) /resOOS_olsHAR_CL1.(vars.namesHAR{j}).MSE(wt)) ];

        [table4(5,i+2,j), table4(6,i+2,j)] = tsCWtest(resOOS_olsHAR_CL1.(vars.namesHAR{j}).uhat(:,wt),resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).uhat(:,wt),...
                                              resOOS_olsHAR_CL1.(vars.namesHAR{j}).yhat(:,2,wt), resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).yhat(:,2,wt), vars.fhorizon(j));

    end

% Print results
clear info;
info.cnames = char('HAR','LHAR','PCA-LHAR-En','PCA-LHAR-PMet','PCA-LHAR-NFMet','PCA-LHAR-Agric','PCA-Total');
info.rnames = char('Table4','MSE','HMSE','MZ beta','CT R2','CW test','');
info.fmt    = '%10.4f';
sprintf('\n\n');
mprint(table4(:,:,j),info);

end
%%
% Out-of-sample estimates report

% Window type = expanding
wt = 2;

% Init table
table5 = nan(6,7,3);

% Save results
for j = 1: length(vars.fhorizon)
    table5(1:4,1,j) = [resOOS_olsHAR_CL1.(vars.namesHAR{j}).MSE(wt)...
                       resOOS_olsHAR_CL1.(vars.namesHAR{j}).HRMSE(wt)...
                       resOOS_olsHAR_CL1.(vars.namesHAR{j}).MZ(wt).beta(2)...
                       resOOS_olsHAR_CL1.(vars.namesHAR{j}).MZ(wt).rbar];

    table5(1:4,2,j) = [resOOS_olsHAR_CL1.(vars.namesLHAR{j}).MSE(wt)/resOOS_olsHAR_CL1.(vars.namesHAR{j}).MSE(wt)...
                       resOOS_olsHAR_CL1.(vars.namesLHAR{j}).HRMSE(wt)/resOOS_olsHAR_CL1.(vars.namesHAR{j}).HRMSE(wt)...
                       resOOS_olsHAR_CL1.(vars.namesLHAR{j}).MZ(wt).beta(2)...
                       1-(resOOS_olsHAR_CL1.(vars.namesLHAR{j}).MSE(wt)/resOOS_olsHAR_CL1.(vars.namesHAR{j}).MSE(wt)) ]; %resOOS_olsHAR_CL1.(vars.namesLHAR{j}).MZ(wt).rbar];
    [table5(5,2,j), table5(6,2,j)] = tsCWtest(resOOS_olsHAR_CL1.(vars.namesHAR{j}).uhat(:,wt),resOOS_olsHAR_CL1.(vars.namesLHAR{j}).uhat(:,wt),...
                                              resOOS_olsHAR_CL1.(vars.namesHAR{j}).yhat(:,2,wt), resOOS_olsHAR_CL1.(vars.namesLHAR{j}).yhat(:,2,wt), vars.fhorizon(j));

    for i = 1:length(vars.levHARDataNames)
        table5(1:4,i+2,j) = [resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MSE(wt)/resOOS_olsHAR_CL1.(vars.namesHAR{j}).MSE(wt)...
                             resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).HRMSE(wt)/resOOS_olsHAR_CL1.(vars.namesHAR{j}).HRMSE(wt)...
                             resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MZ(wt).beta(2)...
                             1-(resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).MSE(wt) /resOOS_olsHAR_CL1.(vars.namesHAR{j}).MSE(wt)) ];

        [table5(5,i+2,j), table5(6,i+2,j)] = tsCWtest(resOOS_olsHAR_CL1.(vars.namesHAR{j}).uhat(:,wt),resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).uhat(:,wt),...
                                              resOOS_olsHAR_CL1.(vars.namesHAR{j}).yhat(:,2,wt), resOOS_olsHAR_CL1.([vars.namesLHARPCA{j}   vars.levHARDataNames{i}]).yhat(:,2,wt), vars.fhorizon(j));

    end

% Print results
clear info;
info.cnames = char('HAR','LHAR','PCA-LHAR-En','PCA-LHAR-Met','PCA-LHAR-NFMet','PCA-LHAR-Agric','PCA-Total');
info.rnames = char('Table4','MSE','MAE','MZ beta','CT R2','CW test','');
info.fmt    = '%10.4f';
sprintf('\n\n');
mprint(table4(:,:,j),info);

end