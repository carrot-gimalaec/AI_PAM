import re
from abc import ABC

import torch
from transformers import AutoModel
from transformers import AutoTokenizer
import copy


class TokenGenerator(ABC):
    def __init__(
        self,
        system_prompt: str,
        model_name: str,
        device: str = "cpu",
    ):
        """
        Инициализирует классификатор с нулевым обучением.

        Parameters:
            system_prompt: Системный промпт для модели.
            model_name: Название предобученной модели для загрузки.
            device: Устройство для выполнения вычислений.
        """
        if "Qwen3" not in model_name:
            raise ValueError(
                f"Model {model_name} is not supported. Please use Qwen3 models."
            )

        self.model = AutoModel.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16
            if not (
                device == "cuda"
                and torch.cuda.is_bf16_supported()
                and "AWQ" not in model_name
            )
            else torch.bfloat16,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.system_prompt = system_prompt

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
            [tokenized_prompt["input_ids"], generated_token.unsqueeze(0)], dim=1
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

        logits = outputs.last_hidden_state @ self.model.embed_tokens.weight.T
        token_distribution = torch.nn.functional.softmax(
            logits[:, -1, :] / temperature,
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


class ZeroShotClassifierWithProbs(TokenGenerator):
    def __init__(self, system_prompt: str, model_name: str, device: str = "cpu"):
        super().__init__(
            system_prompt,
            model_name,
            device,
        )
        self.tokens_before_classification = [151668, 271, 151668, 271]

    @torch.inference_mode()
    def classify(
        self,
        prompt: str,
        target_tokens: dict[str, str],
        do_normalization: bool = True,
        debug: bool = False,
        use_chat_template: bool = False,
    ) -> float:
        torch.cuda.empty_cache()
        if isinstance(prompt, list | tuple):
            raise "Batch classification is not supported yet. Please provide a single prompt."

        # get target token id
        target_token_pos_id = self.tokenizer.encode(
            target_tokens["pos"], add_special_tokens=False
        )
        target_token_neg_id = self.tokenizer.encode(
            target_tokens["neg"], add_special_tokens=False
        )
        if len(target_token_neg_id) + len(target_token_pos_id) > 2:
            raise ValueError(
                f"Target tokens len must be 1. For {target_tokens['pos']} - {len(target_token_pos_id)}, {target_tokens['neg']} - {len(target_token_neg_id)}"
            )

        if use_chat_template:
            pos_prob, neg_prob, most_prob_token = self._classify_with_chat_template(
                prompt, target_token_pos_id[0], target_token_neg_id[0], debug
            )
        else:
            pos_prob, neg_prob, most_prob_token = self._classify(
                prompt, target_token_pos_id[0], target_token_neg_id[0], debug
            )

        if debug:
            print(
                "\n---PROBABILITIES: "
                f"{target_tokens['pos']}: {pos_prob}, {target_tokens['neg']}: {neg_prob}"
                f"\nMOST PROBABLE: {
                    (self.tokenizer.decode(most_prob_token).replace('\n', '\\n'),)
                }, {most_prob_token}"
            )

        if do_normalization:
            pos_prob = pos_prob / (pos_prob + neg_prob)

        return pos_prob

    @torch.no_grad()
    def _classify_with_chat_template(
        self,
        prompt: str,
        pos_token_id: int,
        neg_token_id: int,
        debug: bool,
        max_new_tokens: int = 20,
    ) -> tuple[float, float, int]:
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        tokenized_chat = self.tokenizer.apply_chat_template(
            chat, return_tensors="pt", return_dict=True
        ).to(device=self.model.device)

        assistant_tokens = copy.deepcopy(self.tokens_before_classification)

        # Generate the response
        while True:
            # generate head output
            generated_token, token_distribution = self._generate_new_token(
                tokenized_chat, do_sample=False
            )
            max_new_tokens -= 1

            if len(assistant_tokens) == 0 or (max_new_tokens == 0):
                pos_prob = token_distribution[:, pos_token_id].squeeze().cpu().item()
                neg_prob = token_distribution[:, neg_token_id].squeeze().cpu().item()
                token = generated_token.cpu().item()
                return pos_prob, neg_prob, token
            else:
                if (token := generated_token.cpu().item()) in assistant_tokens:
                    assistant_tokens.remove(token)

                if debug:
                    debug_output = f"{self.tokenizer.decode(token)}"
                    debug_output = debug_output.replace("\n", "\\n")
                    print(debug_output, end="")
                self._add_generated_token(tokenized_chat, generated_token)

    @torch.no_grad()
    def _classify(
        self, prompt: str, pos_token_id: int, neg_token_id: int
    ) -> tuple[float, float, int]:
        chat = self.system_prompt + "\n" + prompt
        tokenized_chat = self.tokenizer(
            chat,
            return_tensors="pt",
        ).to(device=self.model.device)

        generated_token, token_distribution = self._generate_new_token(
            tokenized_chat, do_sample=False
        )
        pos_prob = token_distribution[:, pos_token_id].squeeze().cpu().item()
        neg_prob = token_distribution[:, neg_token_id].squeeze().cpu().item()
        token = generated_token.cpu().item()
        return pos_prob, neg_prob, token


class ZeroShotClassifierWithTextOutput(TokenGenerator):
    def __init__(
        self,
        system_prompt: str,
        model_name: str,
        device: str = "cpu",
    ):
        system_prompt += "\n Для вывода готового ответа используй формат <result></result>, например: <result>да</result>"
        super().__init__(system_prompt, model_name, device)
        self.pattern = r"<result>(.*?)</result>"
        self.eos_token_id = self.tokenizer.eos_token_id

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
        torch.cuda.empty_cache()
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
                    f"{self.tokenizer.decode(generated_token.cpu())}(:{generated_token.cpu()})".replace(
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
