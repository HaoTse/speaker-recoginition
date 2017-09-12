function [result] = recognition(model, f, k)
    nl = size(model, 1);
    len = size(f, 2);
    
    for i = 1:nl
        m = model{i, 1};
        v = model{i, 2};
        for j = 1:k
            for l = 1:len
                pd(:, l) = log(pdf('Normal', f(:, l), m(:, j), v(:, j)));
            end
            tmp = sum(pd, 1);
            result_tmp(i, j) = max(tmp);
        end
        result(i) = max(result_tmp(i, :));
    end
    
end

