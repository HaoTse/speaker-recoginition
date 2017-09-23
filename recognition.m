function [result] = recognition(model, f, k)
    nl = size(model, 1);
    len = size(f, 2);
    
    for i = 1:nl
        pd(1:len) = -inf;
        m = model{i, 1};
        v = model{i, 2};
        for l = 1:len
            for j = 1:k
                tmp(:) = log(exp(-0.5 * ((f(:, l) - m(:, j))./v(:, j)).^2) ./ (sqrt(2*pi) .* v(:, j)));
                tmp1 = sum(tmp);
                if sum(tmp) > pd(l)
                    pd(l) = sum(tmp);
                end
            end
        end
        result(i) = sum(pd);
    end
    
end

