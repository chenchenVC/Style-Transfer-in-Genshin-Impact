function example()
    % 示例数据
    C = 3; H = 100; W = 128;
    features = rand(C, H, W);
    img = rand(C, H, W);
    style_layers = [1, 2];
    style_targets = cell(1, length(style_layers));
    style_weights = [0.5, 0.5];
    content_weight = 1;
    content_layer = 1;
    content_target = rand(C, H, W);
    tv_weight = 0.1;
    
    % 计算Gram矩阵
    gram = gram_matrix(features, 1);
    
    % 计算TV损失
    tv_loss_val = tv_loss(img, tv_weight);
    
    % 计算风格损失
    for i = 1:length(style_layers)
        style_targets{i} = gram_matrix(features(:, style_layers(i), :), 0);
    end
    style_loss_val = style_loss(features, style_layers, style_targets, style_weights);
    
    % 计算内容损失
    content_loss_val = content_loss(content_weight, features(:, content_layer, :), content_target);
    
    % 计算总损失
    total_loss = content_loss_val + style_loss_val + tv_loss_val;
    
    % 输出结果
    fprintf('Gram Matrix:\n');
    disp(gram);
    fprintf('TV Loss: %.4f\n', tv_loss_val);
    fprintf('Style Loss: %.4f\n', style_loss_val);
    fprintf('Content Loss: %.4f\n', content_loss_val);
    fprintf('Total Loss: %.4f\n', total_loss);
end

% 定义函数
function result = gram_matrix(features, normalize)
    [C, H, W] = size(features);
    first = reshape(features, [C, H*W]);
    result = first * first';
    if normalize
        result = result / (H*W*C);
    end
end

function loss = tv_loss(img, tv_weight)
    up = img(:, 1:end-1, :);
    down = img(:, 2:end, :);
    left = img(:, :, 1:end-1);
    right = img(:, :, 2:end);
    first_sum = sum(sum((up - down).^2));
    second_sum = sum(sum((left - right).^2));
    loss = tv_weight * (sum(first_sum) + sum(second_sum));
end

function style_losses = style_loss(feats, style_layers, style_targets, style_weights)
    style_losses = 0;
    for i = 1:length(style_layers)
        now = gram_matrix(feats(:, style_layers(i), :),1) - style_targets{i};
        style_losses = style_losses + style_weights(i) * sum(now(:).*now(:));
    end
end

function loss = content_loss(content_weight, content_current, content_original)
    diffmat = content_current - content_original;
    loss = content_weight * sum(diffmat(:).*diffmat(:));
end