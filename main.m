addpath(genpath('rastamat'));

% Clean-up MATLAB's environment
clear all; close all; clc;

col_header = ['  ', ' S1', ' S2', ' S3', ' S4', ' S5', ' S6'];
row_header = ['S1'; 'S2'; 'S3'; 'S4'; 'S5'; 'S6'];

for fold = 1:5
    for k = 5:10
        for s = 1:6
            %% extract features of all audio
            features = {};
            feature_point = [];
            for i = 0:5
                for j = 1:5
                    file_name = sprintf('record/S%d/%d_%d.wav', s, i, j);
                    feature = mfcc(file_name);
                    features{i * 5 + j} = feature;
                    if j == fold
                        test_features{s, i + 1} = feature;
                        continue;
                    end
                    tmp = j;
                    if j > fold
                        tmp = j - fold;
                    end
                    train_features{s, i * 4 + tmp} = feature;
                    
                    if isempty(feature_point)
                        feature_point = feature;
                    else
                        feature_point = cat(2, feature_point, feature);
                    end
                end
            end

            %% kmeans
            sumd = inf;
            for i = 1:15
                [tmp_clusters, tmp_c, tmp_sumd] = kmeans(feature_point', k);
                tmp_clusters = tmp_clusters';
                if tmp_sumd < sumd
                    clusters = tmp_clusters;
                    sumd = tmp_sumd;
                    c = tmp_c;
                end
            end
            
            %% compute Gaussian model
            for i = 1:k
                % compute mean
                for j = 1:39
                    mean(j, i) = sum(feature_point(j, find(clusters == i)));
                end
                mean(:, i) = mean(:, i) / length(find(clusters == i));

                % compute variance
                for j = 1:39
                    var(j, i) = sum((feature_point(j, find(clusters == i)) - mean(j, i)) .^ 2);
                end
                var(:, i) = sqrt(var(:, i) / length(find(clusters == i)));
            end
            model{s, 1} = mean;
            model{s, 2} = var;
        end

        %% test recognition result
        right = zeros(6, 1);
        total = zeros(6, 1);
        con_matrix = zeros(6, 6);
        for s = 1:6
            for t = 1:6
                test = test_features{s, t};
                re = recognition(model, test, k);
                idx = find(re == max(re), 1);
                right(s) = right(s) + (idx == s);
                total(s) = total(s) + 1;
                result(s, t) = idx;
                con_matrix(s, idx) = con_matrix(s, idx) + 1;
            end
            con_matrix(s, 7) = right(s);
            con_matrix(s, 8) = total(s);
        end
        pro = sum(right) / sum(total);
    
        %% test output result
        filename = sprintf('result/test/fold%d/k%d.txt', fold, k);
        fout = fopen(filename, 'w');
        fprintf(fout, sprintf('probability: %7.4f (%d / %d)', pro, sum(right), sum(total)));
        fclose(fout);
        % output confusion matrix
        filename = sprintf('result/test/fold%d/matrix%d.txt', fold, k);
        fout = fopen(filename, 'w');
        fprintf(fout, col_header);
        fprintf(fout, '\n');
        format = '%3d%3d%3d%3d%3d%3d (%2d/%2d)\n';
        [nrows, ncols] = size(con_matrix);
        for row = 1:nrows
            fprintf(fout, row_header(row, :));
            fprintf(fout, format, con_matrix(row, :));
        end
        fclose(fout);
        
        %% train recognition result
        right = zeros(6, 1);
        total = zeros(6, 1);
        con_matrix = zeros(6, 6);
        for s = 1:6
            for t = 1:24
                test = train_features{s, t};
                re = recognition(model, test, k);
                idx = find(re == max(re), 1);
                right(s) = right(s) + (idx == s);
                total(s) = total(s) + 1;
                result(s, t) = idx;
                con_matrix(s, idx) = con_matrix(s, idx) + 1;
            end
            con_matrix(s, 7) = right(s);
            con_matrix(s, 8) = total(s);
        end
        pro = sum(right) / sum(total);
    
        %% train output result
        filename = sprintf('result/train/fold%d/k%d.txt', fold, k);
        fout = fopen(filename, 'w');
        fprintf(fout, sprintf('probability: %7.4f (%d / %d)', pro, sum(right), sum(total)));
        fclose(fout);
        % output confusion matrix
        filename = sprintf('result/train/fold%d/matrix%d.txt', fold, k);
        fout = fopen(filename, 'w');
        fprintf(fout, col_header);
        fprintf(fout, '\n');
        format = '%3d%3d%3d%3d%3d%3d (%2d/%2d)\n';
        [nrows, ncols] = size(con_matrix);
        for row = 1:nrows
            fprintf(fout, row_header(row, :));
            fprintf(fout, format, con_matrix(row, :));
        end
        fclose(fout);

    end
end