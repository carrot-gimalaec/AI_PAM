import re
from abc import ABC

import torch
from transformers import AutoModel
from transformers import AutoTokenizer


class ZeroShotClassifierInterface(ABC):
    def __init__(
        self,
        system_prompt: str,
        model_name: str = "RefalMachine/RuadaptQwen2.5-1.5B-instruct",
        device: str = "cpu",
    ):
        """
        Инициализирует классификатор с нулевым обучением.

        Parameters:
            system_prompt: Системный промпт для модели.
            model_name: Название предобученной модели для загрузки.
            device: Устройство для выполнения вычислений.
        """
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16,
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

    @staticmethod
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
            [
                tokenized_prompt["attention_mask"],
                torch.ones((1, 1), dtype=torch.int64, device=generated_token.device),
            ],
            dim=1,
        )

    def _generate_new_token(
        self,
        tokenized_prompt: dict[str, torch.Tensor],
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Генерирует новый токен на основе токенизированного промпта.

        Parameters:
            tokenized_prompt: Токенизированный промпт с ключами 'input_ids' и 'attention_mask'.
            do_sample: Использовать ли выборку при генерации токена.
            temperature: Температура для управления случайностью генерации.
            top_p: Параметр для ядерной выборки.

        Returns:
            Сгенерированный токен и распределение токенов.
        """
        # generate new token
        outputs = self.model(**tokenized_prompt)
        token_distribution = torch.nn.functional.softmax(
            (outputs.last_hidden_state @ self.model.embed_tokens.weight.T)[:, -1, :]
            / temperature,
            dim=-1,
        )

        if do_sample:
            sorted_probs, sorted_indices = torch.sort(
                token_distribution, dim=-1, descending=True
            )
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            nucleus_mask = (cumulative_probs < top_p).float()
            nucleus_mask = torch.cat(
                [torch.ones_like(nucleus_mask[:, :1]), nucleus_mask[:, 1:]], dim=1
            )
            sorted_probs = sorted_probs * nucleus_mask
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            probs_idx = torch.multinomial(sorted_probs, num_samples=1)
            generated_token = sorted_indices.gather(1, probs_idx).squeeze()
        else:
            generated_token = torch.argmax(token_distribution, dim=-1)
        return generated_token, token_distribution


class ZeroShotClassifierWithProbs(ZeroShotClassifierInterface):
    @torch.inference_mode()
    def classify(
        self,
        prompt: str,
        target_tokens: dict[str, str],
        do_normalization: bool = True,
        debug: bool = False,
    ) -> float:
        """ """
        if isinstance(prompt, list | tuple):
            raise "Batch classification is not supported yet. Please provide a single prompt."

        # get target token id
        target_token_pos_id = self.tokenizer.encode(
            target_tokens["pos"], add_special_tokens=False
        )
        target_token_neg_id = self.tokenizer.encode(
            target_tokens["neg"], add_special_tokens=False
        )

        # tokenize prompt as a chat
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        tokenized_chat = self.tokenizer.apply_chat_template(
            chat, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        # Generate the response
        while True:
            # generate head output
            generated_token, token_distribution = self._generate_new_token(
                tokenized_chat, do_sample=False
            )
            if debug:
                print(
                    f"{self.tokenizer.decode(generated_token)}".replace("\n", "\\n"),
                    end="",
                )

            # check token
            if generated_token not in self.assistant_prefix_token_ids:
                pos_prob = (
                    token_distribution[:, target_token_pos_id].squeeze().cpu().item()
                )
                neg_prob = (
                    token_distribution[:, target_token_neg_id].squeeze().cpu().item()
                )
                if debug:
                    print(
                        f"{target_tokens['pos']}: {pos_prob}, {target_tokens['neg']}: {neg_prob}"
                    )
                # normalize probabilities
                if do_normalization:
                    pos_prob = pos_prob / (pos_prob + neg_prob)
                return pos_prob
            else:
                self._add_generated_token(tokenized_chat, generated_token)
                continue


class ZeroShotClassifierWithTextOutput(ZeroShotClassifierInterface):
    def __init__(
        self,
        system_prompt: str,
        model_name: str = "RefalMachine/RuadaptQwen2.5-1.5B-instruct",
        device: str = "cpu",
    ):
        system_prompt += "\n Для вывода готового ответа используй формат <result></result>, например: <result>да</result>"
        super().__init__(system_prompt, model_name, device)
        self.pattern = r"<result>(.*?)</result>"

    @torch.inference_mode()
    def answer(
        self,
        prompt: str,
        max_new_tokes: int = 512,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
        debug: bool = False,
    ) -> str:
        # tokenize prompt as a chat
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        tokenized_chat = self.tokenizer.apply_chat_template(
            chat, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        # Generate the response
        total_generated_tokens = 0
        while True:
            # generate head output
            generated_token, _ = self._generate_new_token(
                tokenized_chat,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            if debug:
                print(
                    f"{self.tokenizer.decode(generated_token.cpu())}".replace(
                        "\n", "\\n"
                    ),
                    end="",
                )

            self._add_generated_token(tokenized_chat, generated_token)
            total_generated_tokens += 1

            # check token
            if (
                generated_token == self.eos_token_id
                or total_generated_tokens == max_new_tokes
            ):
                result = self.tokenizer.decode(
                    tokenized_chat["input_ids"][0, -total_generated_tokens:]
                )
                match = re.search(self.pattern, result)
                if match is None:
                    return "Не удалось найти ответ в формате <result></result>."
                else:
                    return match.group(1)
