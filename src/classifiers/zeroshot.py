import numpy as np
import torch
from transformers import AutoModel
from transformers import AutoTokenizer


class ZeroShotClassifier:
    """
    Классификатор для текстовой классификации с использованием предобученной языковой модели.

    Использует предобученную модель трансформера для классификации текста на основе вероятностей целевых токенов.
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "RefalMachine/RuadaptQwen2.5-1.5B-instruct",
        device_map: str = "cpu",
    ):
        """
        Инициализирует классификатор с нулевым обучением.

        Parameters:
            system_prompt: Системный промпт для модели.
            model_name: Название предобученной модели для загрузки.
            device_map: Устройство для выполнения вычислений.
        """
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map=device_map,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.system_prompt = system_prompt

        # assistant prefix tokens, i.e. "<|im_start|>assistant\n"
        self.assistant_prefix_token_ids = [
            torch.tensor(145110),
            torch.tensor(77091),
            torch.tensor(198),
        ]
        self.eos_token_id = self.tokenizer.eos_token_id

    @torch.inference_mode()
    def classify(
        self,
        prompt: str,
        target_tokens: dict[str, str],
        classification_trigger_tokens: str | None = None,
        debug: bool = False,
    ) -> dict[str, float]:
        """
        Классифицирует заданный промпт.

        Parameters:
            prompt: Текст для классификации.
            target_tokens: Словарь с ключами 'pos' и 'neg', содержащий токены для положительного и отрицательного классов.
            classification_trigger_tokens: Токен, который запускает классификацию. Если None, классификация происходит после
                                        первого сгенерированного токена, не являющегося префиксом ассистента.
            debug: Флаг отладки для вывода дополнительной информации.

        Returns:
            Словарь с вероятностями для положительного и отрицательного классов.
        """
        if isinstance(prompt, list | tuple):
            raise "Batch classification is not supported yet. Please provide a single prompt."

        # get target token id
        target_token_pos_id = self.tokenizer.encode(
            target_tokens["pos"], add_special_tokens=False
        )
        target_token_neg_id = self.tokenizer.encode(
            target_tokens["neg"], add_special_tokens=False
        )
        if len(target_token_pos_id + target_token_neg_id) > 2:
            raise ValueError(
                f"Target tokens pos: '{target_tokens['pos']}', neg: '{target_tokens['neg']}' must be a single token. "
                f"Got pos: {len(target_token_pos_id)}, neg: {len(target_token_neg_id)} tokens len instead."
            )
        target_token_pos_id = target_token_pos_id[0]
        target_token_neg_id = target_token_neg_id[0]

        # get classification trigger token id if provided
        if classification_trigger_tokens is not None:
            classification_trigger_token_id = self.tokenizer.encode(
                classification_trigger_tokens,
                add_special_tokens=False,
                return_tensors="np",
            )
        else:
            classification_trigger_token_id = None

        # tokenize prompt as a chat
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        tokenized_chat = self.tokenizer.apply_chat_template(
            chat, return_tensors="pt", return_dict=True
        )

        # Generate the response
        while True:
            # generate head output
            outputs = self.model(**tokenized_chat)
            # multiply the last hidden state with the token embedding matrix and apply softmax to get the token distribution
            token_distribution = torch.nn.functional.softmax(
                torch.matmul(
                    outputs.last_hidden_state, self.model.embed_tokens.weight.T
                )[:, -1, :],
                dim=-1,
            )

            # get the most probable token
            generated_token = torch.argmax(token_distribution, dim=-1)
            if debug:
                print(f"{self.tokenizer.decode(generated_token)}")

            # check token
            if (
                classification_trigger_token_id is None
                and generated_token not in self.assistant_prefix_token_ids
            ):
                pos_prob = (
                    token_distribution[:, target_token_pos_id].squeeze().cpu().item()
                )
                neg_prob = (
                    token_distribution[:, target_token_neg_id].squeeze().cpu().item()
                )
                return {"pos": round(pos_prob, 5), "neg": round(neg_prob, 5)}
            elif classification_trigger_token_id is not None:
                if generated_token == self.eos_token_id:
                    print("End of sequence token detected. Stopping generation.")
                    return {"pos": 0, "neg": 0}
                elif np.all(
                    tokenized_chat["input_ids"][
                        0, -classification_trigger_token_id.shape[1] :
                    ]
                    .cpu()
                    .numpy()
                    == classification_trigger_token_id[0]
                ):
                    pos_prob = (
                        token_distribution[:, target_token_pos_id]
                        .squeeze()
                        .cpu()
                        .item()
                    )
                    neg_prob = (
                        token_distribution[:, target_token_neg_id]
                        .squeeze()
                        .cpu()
                        .item()
                    )
                    return {"pos": round(pos_prob, 5), "neg": round(neg_prob, 5)}
                else:
                    self._add_generated_token(tokenized_chat, generated_token)
                    continue
            else:
                self._add_generated_token(tokenized_chat, generated_token)
                continue

    @staticmethod
    @torch.inference_mode()
    def _add_generated_token(
        tokenized_prompt: dict[str, torch.Tensor], generated_token: torch.Tensor
    ) -> None:
        """
        Добавляет сгенерированный токен к токенизированному промпту.

        Parameters:
            tokenized_prompt: Токенизированный промпт с ключами 'input_ids' и 'attention_mask'.
            generated_token: Сгенерированный токен для добавления.
        """
        tokenized_prompt["input_ids"] = torch.cat(
            [tokenized_prompt["input_ids"], generated_token.view(1, -1)], dim=1
        )
        tokenized_prompt["attention_mask"] = torch.cat(
            [tokenized_prompt["attention_mask"], torch.ones((1, 1), dtype=torch.int64)],
            dim=1,
        )
