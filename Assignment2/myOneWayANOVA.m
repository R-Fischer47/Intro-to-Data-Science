function F = myOneWayANOVA(IV,DV)
    uniqueDVs = unique(DV);
    k = length(uniqueDVs);
    N = length(DV);
    
    %error handeling
    if k == 1
        error("DV has only one unique value")
    elseif length(IV) ~= N
        error("DV does not have the same length as IV")
    end
    

    %group the IV's per category
    %and calculate length, mean, std and var
    for c = 1 : k
        ivs = IV(DV==uniqueDVs(c));
        n(c) = length(ivs);
        meanPerCat(c) = mean(ivs);
        stdPerCat(c) = std(ivs);
        varPerCat(c) = var(ivs);
    end

    globalMean = sum(meanPerCat.*n)/N;
    SSB=sum(((meanPerCat-globalMean).^2).*n)/(k-1);
    SSW = sum(varPerCat.*(n-1))/sum(n-1);
    F = SSB/SSW;
end
