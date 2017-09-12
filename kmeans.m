function [clusters, c, iter] = kmeans(f, k, maxIter)
    if nargin < 3
        maxIter = 30;
    end
    if nargin < 2
        k = 1;
    end
    
    [max_row, max_col] = size(f);
    
    % get random centers
    c = zeros(max_row, k);
    p = randperm(max_col);
    for i = 1:k
        c(:, i) = f(:, p(i));
    end
    
    iter = 0;
    old_rss_error = inf;
    while 1
        for i = 1:max_col
            min_idx = 1;
            min_val = norm(f(:, i) - c(:, 1));
            for j = 2:k
                val = norm(f(:, i) - c(:, j));
                if val < min_val
                    min_idx = j;
                    min_val = val;
                end
            end
            % assign point to clostest center
            clusters(i) = min_idx;
        end
        
        old_c = c;
        % compute new centers
        for i = 1:k
            tmp = f(:, find(clusters == i));
            for j = 1:max_row
                c(j, i) = sum(tmp(j, :));
            end
            c(:, i) = c(:, i) / length(find(clusters == i));
        end
        
        iter = iter + 1;
        
        rss_error = 0;
        for i = 1:max_col
            rss_error = rss_error + norm(f(:, i) - c(:, clusters(i)));
        end
        rss_error = rss_error / max_col;
        
        if rss_error >= old_rss_error
            c = old_c;
            break;
        end
        old_rss_error = rss_error;
        
        if iter > maxIter
            break;
        end
    end
    
end
