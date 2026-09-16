function [] = pocket_perceptron(input)

    % Δεδομένα εισόδου
    x1 = input(:,1); % Χαρακτηριστικό x1
    x2 = input(:,2); % Χαρακτηριστικό x2
    target = input(:,3); % Στόχοι (-1 ή 1)

    % Αρχικοποίηση
    num_samples = length(input); % Αριθμός δειγμάτων
    learning_rate = 0.01; % Ρυθμός εκμάθησης
    max_epochs = 1000; % Μέγιστος αριθμός επαναλήψεων (epochs)
    bias = 1; % Bias
    weights = [0; 0; 0]; % Αρχικοποίηση βαρών σε 0
    best_weights = weights; % Καλύτερα βάρη (Pocket Weights)
    max_correct = 0; % Μέγιστος αριθμός σωστών προβλέψεων

    % Δημιουργία figure για σχεδίαση
    figure('Name', 'Pocket Perceptron');
    hold on;
    grid on;
    class_minus = scatter(x1(target == -1), x2(target == -1), 120, 'r', 'x', 'LineWidth', 2); % Κατηγορία -1
    class_plus = scatter(x1(target == 1), x2(target == 1), 120, 'b', 'o', 'LineWidth', 2); % Κατηγορία 1
    axis([min(x1)-1, max(x1)+1, min(x2)-1, max(x2)+1]); % Ρυθμίσεις άξονα

    % Εκπαίδευση Pocket Perceptron
    for epoch = 1:max_epochs
        correct_predictions = 0; % Καταμέτρηση σωστών προβλέψεων

        for i = 1:num_samples
            % Υπολογισμός πρόβλεψης
            weighted_sum = (bias * weights(3)) + (x1(i) * weights(1)) + (x2(i) * weights(2));
            prediction = sign(weighted_sum); % Υπολογισμός πρόβλεψης

            % Αντιμετώπιση του 0
            if prediction == 0
                prediction = -1;
            end

            % Υπολογισμός σφάλματος
            error = target(i) - prediction;

            % Αν πρόβλεψη σωστή
            if error == 0
                correct_predictions = correct_predictions + 1;
                if correct_predictions > max_correct
                    max_correct = correct_predictions;
                    best_weights = weights; % Αποθήκευση καλύτερων βαρών
                end
            else
                % Ενημέρωση βαρών αν λάθος πρόβλεψη
                weights(1) = weights(1) + learning_rate * x1(i) * error;
                weights(2) = weights(2) + learning_rate * x2(i) * error;
                weights(3) = weights(3) + learning_rate * bias * error;
            end
        end

        % Σχεδίαση διαχωριστικής γραμμής
        if weights(2) ~= 0
            line_x = [min(x1), max(x1)];
            line_y = -((weights(1) * line_x + bias * weights(3)) / weights(2));
            plot(line_x, line_y, 'k--'); % Μαύρη διακεκομμένη γραμμή για τρέχουσα γραμμή
        else
            x_vertical = -weights(3) / weights(1);
            plot([x_vertical, x_vertical], [min(x2)-1, max(x2)+1], 'k--'); % Κατακόρυφη γραμμή
        end
        drawnow;

        % Τερματισμός αν όλες οι προβλέψεις είναι σωστές
        if correct_predictions == num_samples
            fprintf('Όλες οι προβλέψεις είναι σωστές στο epoch %d.\n', epoch);
            break;
        end
    end

    % Σχεδίαση τελικής διαχωριστικής γραμμής
    if best_weights(2) ~= 0
        final_line_y = -((best_weights(1) * line_x + bias * best_weights(3)) / best_weights(2));
        final_line = plot(line_x, final_line_y, 'r-', 'LineWidth', 2); % Κόκκινη γραμμή για τελική διαχωριστική γραμμή
    else
        x_vertical_final = -best_weights(3) / best_weights(1);
        final_line = plot([x_vertical_final, x_vertical_final], [min(x2)-1, max(x2)+1], 'r-', 'LineWidth', 2); % Κατακόρυφη τελική γραμμή
    end

    % Δημιουργία legend
    legend([class_minus, class_plus, final_line], ...
           {'Class -1', 'Class 1', 'Final Decision Boundary'}, ...
           'Location', 'best');

    % Εμφάνιση αποτελεσμάτων
    fprintf('Καλύτερα βάρη (Pocket Weights):\n');
    fprintf('w1: %.4f, w2: %.4f, Bias: %.4f\n', best_weights(1), best_weights(2), best_weights(3));
    fprintf('Μέγιστος αριθμός σωστών προβλέψεων: %d/%d\n', max_correct, num_samples);

end
