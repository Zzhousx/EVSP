function [time, ACC,NMI] = EVSP(train_F,train_L,maxFES,sizep,alpha,beta)                                    
    c= length(unique(train_L));
    %% initializaition similarity
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.WeightMode = 'HeatKernel';
    options.t = 10;
    S = constructW(train_F,options);
    S=(S+S')./2;
    D =diag(sum(S));
    L=D-S;
    %%
    FES = 1;
    dim = size(train_F,2);
    ofit = zeros(sizep,2);
    initThres = 1;
    thres = 0.1; %Exponential decay constant
    paretoAVE = zeros(1,2); %to save final result of the Pareto front
    W0=cell(1,sizep);
    MW=cell(1,sizep);
    %% initializaition
    TDec    = []; 
    Tobj = []; 
    TMask   = [];
    TempPop = [];
    dimFitness = zeros(1,dim);
    for i = 1 : 1
        Dec = ones(dim, dim);
        Mask = eye(dim);
        pop = Dec.*Mask;
        TDec = [TDec;Dec];
        TMask      = [TMask;Mask];
        TempPop    = [TempPop;pop];
        dimfit = zeros(dim, 2);
        for m=1:dim
            [dimfit(m,1),dimfit(m,2)] = FSKNN(pop(m, :),train_F,train_L);
        end
        ofit = fliplr(ofit);
        Tobj = [Tobj; ofit];
        dimFitness    = dimFitness + NDSort(dimfit, dim);  %the order of the Pareto front is used as Fintess 
    end
    %  initializaition the population
    Dec = ones(sizep, dim);
    Mask = zeros(sizep,dim);
    for i = 1 : sizep
        Mask(i, TournamentSelection(2, ceil(rand*dim*0.3), dimFitness)) = 1; 
    end
    off = logical(Dec.*Mask);
    m=c*2; 
    d=dim;
    n=size(train_F,1);
    I=eye(n);
    X=train_F';
    Ln_int=alpha*inv(L+alpha*I);
    M_int=alpha*X*(Ln_int-I)*X';
    M_int=(M_int+M_int')/2;
    em=eig(M_int);
    eta=sort(em);
    if eta(1)<0
        M_int=M_int-(eta(1))*eye(d)+eps*eye(d);
    end
    for i=1:sizep
        Ln{i}=Ln_int;
        M{i}=M_int;
    end
    for i=1:sizep  
        rr=0;
        while rr==0
            W0{i}=rand(d,m);
            if rank(W0{i})==m
                rr=1;
            else
                rr=0;
            end
        end
    end
    inmodel=zeros(sizep,d);
    for i=1:sizep
        k=sum(off(i, :));
        id=randi(d,[1,d-k]);
        W0{i}(id,:)=0;
        W0{i}=orth(W0{i});
        pinvAW = pinv(W0{i}'*M{i}*W0{i});
        P = M{i}*W0{i}*pinvAW*W0{i}'*M{i};
        diagP = diag(P);
        
        [~,index] = sort(diagP,'descend');
        indexW = index(1:k);
        indexO = index(k+1:end);
        for ii=1:k
            j=indexW(ii);
            inmodel(i,j)=true;
        end
        MW{i}=M{i}*W0{i};
        F{i}=Ln{i}*X'*W0{i}; 
        off(i,:)=logical(inmodel(i,:));
        [ofit(i,1),ofit(i,2),ofit(i,3),imp(i,:),W0{i}] = moev(off(i, :),X,MW{i},c,W0{i},indexO,train_L);
    end
    [FrontNO,~] = NDSort(ofit(:,1:2),sizep);
    site = find(FrontNO==1);
    best_fitness=1;
    tic   
    while FES <= maxFES
        isChange = zeros(sizep,dim); 
        %dimensionality reduction
        for i = 1:sizep
            if(ismember(i,site)) 
                continue;
            end
            curiOff = off(i,:); 
            curpSite = site(randi(length(site))); 
            pop = off(curpSite,:); 
            aveiBit = mean(imp(i,:));
            for j = 1:dim
                popBit = pop(j);
                ext = 1/(1+exp(-sizep*(aveiBit-imp(i,j))));
                tempThres = initThres * exp(-thres*FES);
                ext = ext * tempThres*2;
                if rand() < ext
                    off(i,j) = 0; 
                end    
            end
            for j = 1:dim
          %individual repairing
                if imp(i,j) > imp(curpSite,j)
                    off(i,j) = curiOff(j); 
                else 
                    if rand() < (imp(curpSite,j) - imp(i,j))/imp(curpSite,j)
                        off(i,j) = popBit;
                    end
                end
                if curiOff(j) ~= off(i,j)
                    isChange(i,j)=1;
                end
            end
        end
        %evaluate
        for i=1:sizep
            [ofit(i,1),ofit(i,2),ofit(i,3),imp(i,:),W0{i},fs{i}] = moev(off(i, :),X,MW{i},c,W0{i},indexO,train_L);
        end
        [FrontNO,~] = NDSort(ofit(:,1:2),sizep);
        site = find(FrontNO==1);
        solution = ofit(site,:);
        paretoAVE(1) = mean(solution(:,1));
        paretoAVE(2) = mean(solution(:,2));
        paretoAVE(3) = mean(solution(:,3));
        result_site = site;
        result_W0 = W0;  
        if FES==1
            best_fitness=paretoAVE(1);
        end
        if paretoAVE(1) < best_fitness
            best_fitness = paretoAVE(1);
        end 
       length_site_new=length(site);
       iii=1;
       for ii=1:sizep
           if ii~=site(iii) 
                F{ii}=Ln{ii}*X'*W0{ii};    
                for i=1:n 
                    for j=1:n
                        S(i,j)=exp(-(norm(F{ii}(i,:)-F{ii}(j,:),2)^2)/(2*beta)); 
                    end
                    S(i,:)=S(i,:)./sum(S(i,:));
                end
                D = zeros(n, n);
                for i = 1:n
                     D(i, :) = sqrt(sum((F{ii}(i, :) - F{ii}) .^ 2, 2))';
                end
                S = exp(-(D .^ 2) / (2 * beta));
                S = S ./ sum(S, 2);
                S=(S+S')./2;
                D=diag(sum(S,2));
                L=D-S;
                Ln{ii}=alpha*inv(L+alpha*I);
                M{ii}=alpha*X*(Ln{ii}-I)*X';
                M{ii}=(M{ii}+M{ii}')/2;
                MW{ii}=M{ii}*W0{ii}; 
           else
               if site(iii) ~= site(length_site_new)
                   iii=iii+1;
               end
           end
       end
        FES = FES + 1;   
    end
    toc
    %%
    size_site = length(result_site);
    jj=1;

    for ii=1:size_site
        result_SS=sum(result_W0{result_site(jj)}'.*result_W0{result_site(jj)}');
        [~,S]=sort(result_SS,'descend');
        fs_new=round(fs{jj});
        Xw = X(S(1:fs_new),:);
        for i=1:10
            label=kmeans(Xw',c,'maxIter',100,'replicates',20,'EmptyAction','singleton');
            result1 = ClusteringMeasure(train_L,label); 
            result(i,:) = result1;
        end
        for j=1:2
             a=result(:,j);
             ll=length(a);
             temp=[];
             for i=1:ll
                 if i<ll-8
                     b=sum(a(i:i+9));
                     temp=[temp;b];
                 end
             end
            [e,f]=max(temp);
            e=e./10;
            MEAN(j,:)=[e,f];
            STD(j,:)=std(result(f:f+9,j));
         end
        result_ACC(ii)=MEAN(1,1);
        result_NMI(ii)=MEAN(2,1);
        result_ACC_STD(ii)=STD(1,1);
        result_NMI_STD(ii)=STD(2,1);
        jj=jj+1;
    end
    [~,best_index]=sort(result_ACC,'descend');
    ACC=result_ACC(best_index(1));
    NMI=result_NMI(best_index(1));
    ACC_STD=result_ACC_STD(best_index(1));
    NMI_STD=result_NMI_STD(best_index(1));
    fprintf('ACC: %.5f NMI: %.5f ',ACC,NMI);
    time = toc;
end