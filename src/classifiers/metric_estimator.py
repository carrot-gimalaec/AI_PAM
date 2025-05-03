import plotly.express as px
import plotly.graph_objects as go
import torch


class ClassificationMetricEstimator:
    @staticmethod
    def recall(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Преобразуем тензоры предсказаний и целевых значений в булевы тензоры
        pred = pred.bool()
        target = target.bool()

        # Вычисляем True Positive (TP) - количество правильно предсказанных положительных примеров
        tp = (pred * target).sum()
        # Вычисляем False Negative (FN) - количество положительных примеров, неправильно предсказанных как отрицательные
        fn = (~pred * target).sum()
        # Вычисляем полноту (Recall) = TP / (TP + FN)
        # Добавляем 1e-8 для предотвращения деления на ноль
        return tp / (tp + fn + 1e-8)

    @staticmethod
    def precision(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Преобразуем тензоры предсказаний и целевых значений в булевы тензоры
        pred = pred.bool()
        target = target.bool()

        # Вычисляем True Positive (TP) - количество правильно предсказанных положительных примеров
        tp = (pred * target).sum()
        # Вычисляем False Positive (FP) - количество отрицательных примеров, неправильно предсказанных как положительные
        fp = (pred * ~target).sum()
        # Вычисляем точность (Precision) = TP / (TP + FP)
        # Добавляем 1e-8 для предотвращения деления на ноль
        return tp / (tp + fp + 1e-8)

    def f1(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Преобразуем тензоры предсказаний и целевых значений в булевы тензоры
        pred = pred.bool()
        target = target.bool()

        # Вычисляем точность и полноту, используя соответствующие методы класса
        precision = self.precision(pred, target)
        recall = self.recall(pred, target)
        # Вычисляем F1-меру как гармоническое среднее точности и полноты
        # F1 = 2 * (precision * recall) / (precision + recall)
        # Добавляем 1e-8 для предотвращения деления на ноль
        return 2 * (precision * recall) / (precision + recall + 1e-8)

    @staticmethod
    def roc_auc(pred_probs: torch.Tensor, target: torch.Tensor) -> go.Figure:
        # Создаем линейно распределенные пороги от 1 до 0 для построения ROC-кривой
        thresholds = torch.linspace(1, 0, 100)

        # Инициализируем массивы для хранения значений True Positive Rate (TPR) и False Positive Rate (FPR)
        tpr = torch.zeros(thresholds.shape)
        fpr = torch.zeros(thresholds.shape)

        # Для каждого порога вычисляем TPR и FPR
        for i, threshold in enumerate(thresholds):
            # Применяем порог к вероятностям для получения бинарных предсказаний
            pred = torch.where(pred_probs > threshold, 1, 0)

            # Преобразуем тензоры в булевы
            pred = pred.bool()
            target = target.bool()

            # Вычисляем True Positive (TP), False Positive (FP), False Negative (FN), True Negative (TN)
            tp = (pred * target).sum()
            fp = (pred * ~target).sum()
            fn = (~pred * target).sum()
            tn = (~pred * ~target).sum()

            # Вычисляем TPR (Sensitivity, Recall) = TP / (TP + FN)
            tpr[i] = tp / (tp + fn + 1e-8)
            # Вычисляем FPR = FP / (FP + TN)
            fpr[i] = fp / (fp + tn + 1e-8)

        # Вычисляем площадь под ROC-кривой (AUC) методом трапеций
        auc = 0.0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i - 1]) * ((tpr[i] + tpr[i - 1]) / 2)

        # Создаем график ROC-кривой с помощью plotly express
        fig = px.area(
            x=fpr.cpu().numpy(),
            y=tpr.cpu().numpy(),
            title=f"ROC Curve (AUC={auc.item():.4f})",
            labels=dict(x="False Positive Rate", y="True Positive Rate"),
            width=700,
            height=500,
        )
        # Добавляем диагональную линию (baseline для случайного классификатора)
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(color="red", width=2, dash="dash"),
        )
        return fig

    @staticmethod
    def confusion_matrix(pred: torch.Tensor, target: torch.Tensor) -> go.Figure:
        # Преобразуем тензоры предсказаний и целевых значений в булевы тензоры
        pred = pred.bool()
        target = target.bool()

        # Вычисляем True Positive (TP), True Negative (TN), False Negative (FN), False Positive (FP)
        # и преобразуем в скалярные значения для визуализации
        tp = (pred * target).sum().cpu().item()
        tn = (~pred * ~target).sum().cpu().item()
        fn = (~pred * target).sum().cpu().item()
        fp = (pred * ~target).sum().cpu().item()

        # Создаем матрицу ошибок для визуализации
        z = [[tp, fn], [fp, tn]]

        # Метки для осей графика
        x = ["Predicted POS", "Predicted NEG"]
        y = ["Actual POS", "Actual NEG"]

        # Создаем тепловую карту (heatmap) для визуализации матрицы ошибок
        fig = go.Figure(
            data=go.Heatmap(z=z, x=x, y=y, colorscale="Blues", showscale=False)
        )

        # Добавляем аннотации с числовыми значениями на тепловую карту
        annotations = []
        for i in range(len(y)):
            for j in range(len(x)):
                annotations.append(
                    {
                        "x": x[j],
                        "y": y[i],
                        "text": str(z[i][j]),
                        "font": {"color": "black", "size": 14},
                        "showarrow": False,
                    }
                )

        # Обновляем макет графика, добавляем заголовок и подписи осей
        fig.update_layout(
            title="Confusion Matrix",
            annotations=annotations,
            xaxis=dict(title="Predicted"),
            yaxis=dict(title="Actual"),
        )

        return fig
