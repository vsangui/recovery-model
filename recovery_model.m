% Model of neuromotor recovery from robot-assisted training
%
% (C) Vittorio Sanguineti 2012-2015
%
% Casadio M, Sanguineti V. (2012) 
% Learning, retention, and slacking: a model of the dynamics of recovery in robot therapy. 
% IEEE Trans Neural Syst Rehabil Eng. 20(3):286-96. 
% doi: 10.1109/TNSRE.2012.2190827. 
% 
% Uses MATLAB System Identification Toolbox

close all
clear all

%% Dataset definition (one file per subject)

datadir = './dataset/';

% Data file is assumed to be a .mat file with a single struct variable with structure below: 
recnames = {'subject',...  % subject ID
            'session',...  % session no.
            'force',...    % magnitude of assistive force
            'vision',...   % vision of the hand (1: yes, 0: no)
            'target',...   % target no.
            'peaks',...    % number of peaks (a measure of smoothness)
            'speed',...    % average speed
            'rom',...      % distance to target after first sub-movement (endpoint error)
            'totaltime',...% total movement duration
            'time2peak',...% duration of first sub-movement
            'tratio',...   % 
            };    

dd = dir([datadir,'*.mat']);
nsubj = length(dd);

% subjects IDs are taken from filenames
for s = 1:nsubj
    fname = dd(s).name;
    xlab{s}=sprintf('%s',fname(1:(end-4)));
end


for subj = 1:nsubj 
    subjname = xlab{subj};
    subjID = xlab{subj};
    data = load([datadir, subjname]);    
    
    for nrec = 1:length(recnames)
       eval(['d.',recnames{nrec},'=data.outs(:,nrec);']);
    end
    
    % sort time series
    ii = find(d.session >=1 );
    sess = d.session(ii);
    t_init =[1; find([0; diff(sess)])];  % trial no at each beginning of session
    
    % Specify output (performance measure)
    perf = d.speed(ii);
    perf_str = 'SPEED [m/s]';
    vol_str = 'VOLUNTARY [m/s]';
    perf_range = [0 6];
    
    % Specify FORCE (input 1)
    asst = d.force(ii);             % force magnitude
    asst_str = 'FORCE [N]';
    asst_range = [0 25]; 
    
    % Specify VISION (input 2)
    vis = d.vision(ii);          % vision (1) or no-vision (0)
    
    % specify DRIVING SIGNAL 
    rom_max = 0.3; % target distance
    rom_min = 0.02;% target radius
    nendp_error = (d.rom(ii)-rom_min)./(rom_max-rom_min); % normalized endpoint error
    driving  = 1-nendp_error;                             % normalized endpoint performance
    driving_str = 'DRIVING';
    

     
    % Display time series
    figure(subj)
        figname = [subjID,'-model_persession'];
    set(gcf,'pos',[400 100 500 600],'name',figname)

    subplot(4,1,1);  % assistance
        
        % draw shaded vision input 
        sessions = min(sess):max(sess);
        for s = 1:length(sessions)
         session = sessions(s);   
         is = find(sess==session);

         % display vision as a grey (no vision) or a white area (vision)
         i=find(vis(is)==0);
         t_in = is(i(1));
         t_fi = is(i(end));
         hp=patch([ t_in t_in:t_fi t_fi],0.9*min(asst)+[0 1-vis(t_in:t_fi)' 0].*(1.1*max(asst)-0.9*min(asst)),[0.9 0.9 0.9]);
         set(hp,'edgecol',[0.9 0.9 0.9])
        end
        
        for s = 1:length(sessions)
         session = sessions(s);   
         is = find(sess==session);

         line(is,asst(is),'col','k','linew',1)  
         line(is(end)*[1 1],[0.9*min(asst) 1.1*max(asst)],'col','k','lines',':')
        end
        
        set(gca,'ylim',[0.9*min(asst) 1.1*max(asst)],'xlim',[1 length(asst)])
    ylabel(asst_str)

    subplot(4,1,2); % driving signal
       % draw shaded vision input 
        for s = 1:length(sessions)
         session = sessions(s);   
         is = find(sess==session);

         % display vision as a grey (no vision) or a white area (vision)
         i=find(vis(is)==0);
         t_in = is(i(1));
         t_fi = is(i(end));
         hp=patch([ t_in t_in:t_fi t_fi],0.9*min(driving)+[0 1-vis(t_in:t_fi)' 0].*(1.1*max(driving)-0.9*min(driving)),[0.9 0.9 0.9]);
         set(hp,'edgecol',[0.9 0.9 0.9])
        end
        
        for s = 1:length(sessions)
         session = sessions(s);   
         is = find(sess==session);
         line(is,driving(is),'col','k','linew',1)
         line(is(end)*[1 1],[0.9*min(driving) 1.1*max(driving)],'col','k','lines',':')
        end
    ylabel(driving_str)
    set(gca,'ylim',[0.9*min(driving) 1.1*max(driving)],'xlim',[1 length(driving)])

    subplot(4,1,3); % performance
        % draw shaded vision input 
        for s = 1:length(sessions)
         session = sessions(s);   
         is = find(sess==session);

         % display vision as a grey (no vision) or a white area (vision)
         i=find(vis(is)==0);
         t_in = is(i(1));
         t_fi = is(i(end));
         hp=patch([ t_in t_in:t_fi t_fi],0.9*min(perf)+[0 1-vis(t_in:t_fi)' 0].*(1.1*max(perf)-0.9*min(perf)),[0.9 0.9 0.9]);
         set(hp,'edgecol',[0.9 0.9 0.9])
        end
        
        for s = 1:length(sessions)
         session = sessions(s);   
         is = find(sess==session);
         line(is,perf(is),'col','k')
         line(is(end)*[1 1],[0.9*min(perf) 1.1*max(perf)],'col','k','lines',':')
       
        end
    set(gca,'xlim',[1 length(perf)],'ylim',[0.9*min(perf) 1.1*max(perf)])
    ylabel(perf_str)
    
    subplot(4,1,4); % voluntary control

    % draw shaded vision input 
        for s = 1:length(sessions)
         session = sessions(s);   
         is = find(sess==session);

         % display vision as a grey (no vision) or a white area (vision)
         i=find(vis(is)==0);
         t_in = is(i(1));
         t_fi = is(i(end));
         hp=patch([ t_in t_in:t_fi t_fi],(-0.1)+[0 1-vis(t_in:t_fi)' 0].*(0.2+0.1),[0.9 0.9 0.9]);
         set(hp,'edgecol',[0.9 0.9 0.9])
        end

    ylabel(vol_str)
    xlabel('TRIALS')
   drawnow
   
   
    % Define model structure (to be changed according to your hypotheses)
    % 
    %  General form:
    %
    %  y(t) = C x(t) + D [u(t) w(t) ] + r(t)
    %  x(t+1) = A x(t) + B [u(t) w(t)] + q(t)
    %
    %
    %  Predictive form:
    %
    %  x(t+1) = A x(t) + B [u(t) w(t)] + K r(t)
    %
    
    % Step 1: define your inputs and outputs
    Y = perf'; 
    U = [driving'];
    W = [asst'; vis'];
    
    T = length(Y); % timescale
   
    
    % Step 2: define model structure
    As = nan;  % A is free to vary
    Bs = [nan nan nan]; % all B components are free to vary
    Cs = 1;
    Ds = [0 nan nan];
    
    Ks = nan;
    x0s = nan;      
    
    % Step 3: initialise model parameters
    Ac = 0.9;
    Bc = [0.1 0.1 0.1];
    Cc = 1;
    Dc = [0 0.1 0.1];
    Kc = 0.1;
    x0c = 0.1;
    
    % Step 4: Create model object
    m1ini = idss(Ac,Bc,Cc,Dc,Kc,x0c,1);
    setstruc(m1ini,As,Bs,Cs,Ds,Ks,x0s)
    
    % Step 5: subtracting means usually does good to identification
    Ym = mean(Y');
    mmY = Y'-Ym;  
    
    Um = mean(U');
    mmU = U - Um(ones(T,1),:)';
    
    Wm = mean(W');        
    mmW = W - Wm(ones(T,1),:)';
   
    % Step 6: create IDDATA object
    sessions = min(sess):max(sess);
    for s = 1:length(sessions)
        session = sessions(s);   
        is = find(sess==session);
        
        trnset = is;
        valset = is;
        
        % generate multisession inputs
        mY{s} = mmY(is);        
        mWU{s} = [mmU(:,is)' mmW(:,is)'];

        if s==1
          trndata = iddata(mY{s},mWU{s},1);
        else
          trndata = merge(trndata,iddata(mY{s},mWU{s},1));
        end    
    end
    trndata.tstart = num2cell(t_init);
    
    
    % Step 7: Run system identification procedure!
    the_focus ='stability';
    %the_focus ='prediction';
    %the_focus = 'simulation';
    
    % Find model to data...
    m1 = pem(trndata,m1ini,'focus',the_focus,'initialstate','backcast');%,'disturbancemodel','zero');% ,'MaxIter', 1000,'Tolerance',0.000001);%'tolerance',1e-6); % performance-base recovery
    
    % Step 8: predict initial state...
    [e,X01]=pe(m1,trndata,'e'); 
    
    % Step 9: estimate output...
    options = simOptions('InitialCondition',[X01{:}]);
    Yest1 = sim(m1,trndata,options);     
    Ye1= cell2mat(Yest1.OutputData');   
    Ye1_withmean = Ye1+Ym; % don't forget to add mean...
    
    % Step 10: Display model predictions
    subplot(4,1,3)
    line(1:length(Ye1),Ye1_withmean,'col','r') 
    
    % Step 11: Display estimated internal state
    subplot(4,1,4)
    Xe1 = zeros(length(Y),1);
    for s = 1:length(sessions)
        session = sessions(s);   
        is = find(sess==session);
        for t=is'
          if t==is(1)
              Xe1(t) = X01{s};
          else
              Xe1(t) = m1.A*Xe1(t-1) + m1.B*[mmU(t-1);mmW(:,t-1)] + m1.K*(mmY(t-1)-Ye1(t-1));
          end
         line(is(end)*[1 1],[-0.1 0.2],'col','k','lines',':')
     
        end
    end
    Xmean = Ym-m1.D*[Um;Wm'];
    Xe1 = Xe1 + Xmean;
    line(1:length(Y),Xe1,'col','r');
    
    
    
    line(t_init,[X01{:}]+Xmean,'col','r','marker','.','markers',20,'lines','none')
    ylim([-0.1, 0.2])
    xlim([1 length(perf)])
    
    rr=corrcoef(Ye1,Y);
    R2(subj) = rr(1,2).^2
    akaike(subj)= aic(m1);
end % of subject

