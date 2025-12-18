import argparse
import gc
import json
import os
from typing import Any, Dict, Optional

import fitz  # PyMuPDF for PDF processing
import torch  # PyTorch for tensor computations
from tokenizers import Tokenizer  # HuggingFace tokenizers
from semantic_text_splitter import TextSplitter  # Text splitting for large documents

# LangChain imports (keep backwards compatibility across LangChain versions)
try:
    from langchain_community.llms import CTransformers
except Exception:  # pragma: no cover
    from langchain.llms import CTransformers  # type: ignore


class DocumentProcessor:
    """Process a PDF and generate Q&A pairs for fine-tuning.

    Supports two model presets out of the box:
    - Mistral (GGUF via LangChain CTransformers)
    - Qwen (HF Transformers)

    You can override model ids/files via constructor parameters.
    """

    MODEL_PRESETS: Dict[str, Dict[str, str]] = {
        # GGUF preset via CTransformers
        "mistral": {
            "backend": "ctransformers",
            "model_repo": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            "model_file": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "prompt_style": "mistral",
            # HF tokenizer id used only for chunking (best-effort fallback)
            "tokenizer_id": "mistralai/Mistral-7B-Instruct-v0.1",
        },
        # HF Transformers preset
        "qwen": {
            "backend": "transformers",
            "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
            "prompt_style": "qwen",
            # HF tokenizer id used for chunking (best-effort fallback)
            "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
        },
    }

    def __init__(
        self,
        book_path: str,
        temp_folder: str,
        output_file: str,
        book_name: str,
        start_page: int,
        end_page: int,
        number_Q_A: str,
        target_information: str,
        max_new_tokens: int = 500,
        temperature: float = 0.1,
        context_length: int = 1000,
        gpu_layers: int = 100,
        max_tokens_chunk: int = 400,
        arbitrary_prompt: str = "",
        # --- model options ---
        model: str = "mistral",
        llm_backend: Optional[str] = None,
        model_repo: Optional[str] = None,
        model_file: Optional[str] = None,
        hf_model_id: Optional[str] = None,
        system_prompt: str = "You are a helpful, respectful and honest assistant. Answer exactly from the context.",
        prompt_template: Optional[str] = None,
        transformers_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the DocumentProcessor.

        Parameters match the original version, with additional options:

        - model: "mistral" (default) or "qwen"
        - llm_backend: override backend ("ctransformers" or "transformers")
        - model_repo/model_file: overrides for GGUF (ctransformers) loading
        - hf_model_id: override HF model id for transformers backend
        - prompt_template: override prompt formatting
        - transformers_kwargs: passed through to AutoModelForCausalLM.from_pretrained
        """

        self.book_path = book_path
        self.temp_folder = temp_folder
        self.output_file = output_file
        self.books = {book_name: [start_page, end_page]}
        self.number_Q_A = number_Q_A
        self.target_information = target_information
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.context_length = context_length
        self.gpu_layers = gpu_layers
        self.max_tokens_chunk = max_tokens_chunk
        self.arbitrary_prompt = arbitrary_prompt
        self.model = (model or "mistral").lower()

        preset = self.MODEL_PRESETS.get(self.model, self.MODEL_PRESETS["mistral"])
        self.llm_backend = (llm_backend or preset.get("backend", "ctransformers")).lower()

        self.model_repo = model_repo or preset.get("model_repo")
        self.model_file = model_file or preset.get("model_file")
        self.hf_model_id = hf_model_id or preset.get("hf_model_id")
        self.prompt_style = preset.get("prompt_style", "mistral")

        self.system_prompt = system_prompt
        self.transformers_kwargs = transformers_kwargs or {}

        # References for optional prompt-building + cleanup
        self._hf_tokenizer = None  # set for transformers backend
        self._model_ref = None
        self._tokenizer_ref = None

        # Default prompt if arbitrary_prompt is not provided
        if len(self.arbitrary_prompt) > 10:
            self.question_p = self.arbitrary_prompt
        else:
            self.question_p = (
                f"I need you to extract up to {self.number_Q_A} sets of complex questions and their corresponding "
                f"answers from the provided text. The questions and answers should focus on {self.target_information} "
                "and must be based directly on the input text. Please ensure the questions are meaningful and avoid "
                "asking about the main idea or purpose of the text. Do not use pronouns or phrases like 'this period' "
                "and 'in this text' in your questions. Complex questions should require answers that involve two or "
                "more steps. Use specific names and terms for people, locations, agreements, dates, events, and "
                f"{self.target_information} instead of pronouns such as 'he', 'she', 'they', 'him', 'her', 'them', "
                "'this', 'these', or 'those'. For example, ask 'What is the nature of 17th-century society?' instead "
                "of 'What is the nature of this society?' Each question must start with a number (e.g., '1. What is ...'). "
                "Provide each question immediately followed by its answer."
            )

        # Mapping written numbers to numeric equivalents
        self.written_numbers = [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
            "twenty-one",
        ]
        self.number_dict = {self.written_numbers[i]: i + 1 for i in range(len(self.written_numbers))}

        if self.number_Q_A not in self.number_dict:
            raise ValueError(
                f"number_Q_A must be a written number like 'one', 'two', ... up to '{self.written_numbers[-1]}'. "
                f"Got: {self.number_Q_A!r}"
            )

        self.valid_number = [index for index in range(1, self.number_dict[self.number_Q_A] + 1)]

        # Generation configuration for GGUF backend
        self.config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "context_length": self.context_length,
            "gpu_layers": self.gpu_layers,
        }

        # Default prompt templates
        if prompt_template is not None:
            self.template = prompt_template
        else:
            if self.prompt_style == "qwen":
                # Fallback Qwen chat style 
                self.template = (
                    "<|im_start|>system\n{system}\n<|im_end|>\n"
                    "<|im_start|>user\nContext:\n{context}\n\nInstruction:\n{question}\n<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            else:
                # Mistral / LLaMA-style instruct (now also uses system_prompt)
                self.template = (
                    "<s>[INST] {system}\n"
                    "Context:\n{context}\n\n"
                    "Instruction:\n{question} [/INST]"
                )

        # Initialize tokenizer and text splitter (best-effort: match active model)
        tokenizer_id = preset.get("tokenizer_id") or "bert-base-uncased"
        try:
            self.tokenizer22 = Tokenizer.from_pretrained(tokenizer_id)
        except Exception:
            self.tokenizer22 = Tokenizer.from_pretrained("bert-base-uncased")
        self.splitter = TextSplitter.from_huggingface_tokenizer(self.tokenizer22, self.max_tokens_chunk)

        # List to store all prompts
        self.all_prompts = []

    def _build_prompt(self, context: str) -> str:
        # Best practice for Qwen: use the tokenizer chat template (if available)
        if self.prompt_style == "qwen" and self._hf_tokenizer is not None:
            tok = self._hf_tokenizer
            if hasattr(tok, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nInstruction:\n{self.question_p}"},
                ]
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Fallback: string template formatting
        return self.template.format(system=self.system_prompt, context=context, question=self.question_p)

    def _init_llm(self):
        """Create and return the LLM callable based on selected backend."""
        if self.llm_backend == "transformers":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "Transformers backend selected but 'transformers' is not installed. "
                    "Install with: pip install transformers"
                ) from e

            if not self.hf_model_id:
                raise ValueError("hf_model_id is required for transformers backend.")

            kwargs = dict(self.transformers_kwargs)

            # If user does not provide torch_dtype, use fp16 on CUDA when possible
            if "torch_dtype" not in kwargs and torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.float16

            # Prefer device_map=auto if user didn't specify; requires accelerate
            if "device_map" not in kwargs:
                try:
                    import accelerate  # noqa: F401

                    kwargs["device_map"] = "auto"
                except Exception:
                    kwargs["device_map"] = None

            tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id,
                **{k: v for k, v in kwargs.items() if v is not None},
            )
            model.eval()

            # Keep refs for prompt building + cleanup
            self._hf_tokenizer = tokenizer
            self._model_ref = model
            self._tokenizer_ref = tokenizer

            # Build pipeline. If device_map is used, do not pass device.
            pipe_kwargs: Dict[str, Any] = {}
            if kwargs.get("device_map") is None:
                pipe_kwargs["device"] = 0 if torch.cuda.is_available() else -1

            text_gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                pad_token_id=tokenizer.eos_token_id,
                **pipe_kwargs,
            )

            def _call(prompt: str) -> str:
                with torch.inference_mode():
                    out = text_gen(
                        prompt,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=self.temperature > 0,
                        temperature=float(self.temperature),
                        return_full_text=False,
                    )
                if isinstance(out, list) and out and isinstance(out[0], dict):
                    return out[0].get("generated_text", "")
                return str(out)

            return _call

        # Default: GGUF via CTransformers
        if not self.model_repo or not self.model_file:
            raise ValueError("model_repo and model_file are required for ctransformers backend.")

        llm = CTransformers(
            model=self.model_repo,
            model_file=self.model_file,
            config=self.config,
            threads=os.cpu_count() or 1,
        )

        def _call(prompt: str) -> str:
            return llm(prompt)

        # keep a reference for explicit cleanup
        _call._llm_ref = llm  # type: ignore[attr-defined]
        return _call

    def process_book(self):
        """Extract text, split into chunks, and generate Q&A pairs."""
        os.makedirs(self.temp_folder, exist_ok=True)
        out_path = os.path.join(self.temp_folder, list(self.books.keys())[0] + "_QA.txt")

        key = list(self.books.keys())[0]
        with open(out_path, "w", encoding="utf-8") as prompts_write:
            path = os.path.join(self.book_path, key)
            count = 0
            with fitz.open(path) as doc:
                whole_document = ""
                for page in doc:
                    if count >= self.books[key][0] and count < self.books[key][1]:
                        text = page.get_text("blocks")
                        for e in text:
                            whole_document += e[-3].replace("¬\n", "").replace("¬ \n", "")
                    count += 1

                chunks = self.splitter.chunks(whole_document.strip())

            print("chunks: ", len(chunks))

            llm_call = self._init_llm()

            count = 0
            for context in chunks:
                prompt_str = self._build_prompt(context)
                response = llm_call(prompt_str)

                prompts_write.write(
                    "book_name: "
                    + key
                    + " Chunk: "
                    + context.replace("\n", " ")
                    + " \n "
                    + response
                )
                count += 1
                prompts_write.flush()
                prompts_write.write("\n******************************\n")
                print("Chunk ", str(count), " ***done** ")

            # Cleanup
            print("'process_book =_= done '")

            # GGUF cleanup (ctransformers)
            try:
                llm_ref = getattr(llm_call, "_llm_ref", None)
                if llm_ref is not None:
                    del llm_ref
            except Exception:
                pass

            # Transformers cleanup
            try:
                if getattr(self, "_model_ref", None) is not None:
                    del self._model_ref
                if getattr(self, "_tokenizer_ref", None) is not None:
                    del self._tokenizer_ref
                self._hf_tokenizer = None
            except Exception:
                pass

            try:
                del response
            except Exception:
                pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_prompts(self):
        """Generate prompts from the processed Q&A pairs."""
        key = list(self.books.keys())[0]
        in_path = os.path.join(self.temp_folder, key + "_QA.txt")

        Begin = "<s> ### Question: { "

        with open(in_path, "r", encoding="utf-8") as reader:
            flag = True
            Allow = False
            new_line = ""

            for line in reader:
                if "****" not in line and line.rstrip().lstrip().strip() != "\n":
                    if flag:
                        book_name1 = (
                            " the books' name is "
                            + line.split("Chunk:")[0]
                            .split("book_name:")[1]
                            .replace(".pdf", "")
                            .rstrip()
                            .lstrip()
                            .strip()
                        )
                        flag = False

                    if "?" in line:
                        for tt in self.valid_number:
                            if str(tt) + ". " in line:
                                Allow = True
                                break
                            else:
                                Allow = False

                        if Allow:
                            if new_line:
                                self.all_prompts.append(
                                    new_line.rstrip().lstrip().strip().replace("\n", "") + "} </s> \n \n "
                                )

                            new_line = (
                                Begin
                                + line.replace("\n", "")
                                .replace("1.", "")
                                .replace("2.", "")
                                .replace("3.", "")
                                .replace("4.", "")
                                .replace("5.", "")
                                .replace("6.", "")
                                .replace("7.", "")
                                .replace("8.", "")
                                .replace("9.", "")
                                .replace("10.", "")
                                .replace("11.", "")
                                .replace("12.", "")
                                + book_name1
                                + " } ### Answer: { "
                            )
                    else:
                        if Allow:
                            new_line = new_line + line.replace("Answer:", "")
                else:
                    if line.rstrip().lstrip().strip() != "\n":
                        if new_line:
                            self.all_prompts.append(
                                new_line.rstrip().lstrip().strip().replace("\n", "") + "} </s> \n \n "
                            )
                        Allow = False
                        flag = True
                        new_line = ""

        if new_line:
            self.all_prompts.append(new_line.rstrip().lstrip().strip().replace("\n", "") + "} </s> \n \n ")

        print("*************************************")
        for each_prompt in self.all_prompts:
            print(each_prompt)
            print("*************************************")
        print("*** done ***")

    def save_to_jsonl(self):
        """Save the generated prompts to a JSONL file."""
        out_dir = os.path.dirname(os.path.abspath(self.output_file))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.output_file, "w", encoding="latin1") as output_jsonl_file:
            for item in self.all_prompts:
                json_object = {"text": item}
                output_jsonl_file.write(json.dumps(json_object) + "\n")


def main():
    """Minimal CLI entry-point used by `process_document` console script."""
    parser = argparse.ArgumentParser(description="Generate Q&A prompts from a PDF.")
    parser.add_argument("--book_path", required=True, help="Directory containing the PDF file")
    parser.add_argument("--book_name", required=True, help="PDF filename (e.g., example.pdf)")
    parser.add_argument("--temp_folder", required=True, help="Folder for intermediate files")
    parser.add_argument("--output_file", required=True, help="Output JSONL path")
    parser.add_argument("--start_page", type=int, default=0)
    parser.add_argument("--end_page", type=int, default=999999)
    parser.add_argument("--number_Q_A", default="one")
    parser.add_argument(
        "--target_information",
        default="people, dates, agreements, organisations, companies and locations",
    )
    parser.add_argument("--model", default="mistral", choices=sorted(DocumentProcessor.MODEL_PRESETS.keys()))
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--context_length", type=int, default=1000)
    parser.add_argument("--gpu_layers", type=int, default=100)
    parser.add_argument("--max_tokens_chunk", type=int, default=400)
    parser.add_argument("--arbitrary_prompt", default="")

    args = parser.parse_args()

    processor = DocumentProcessor(
        book_path=args.book_path,
        temp_folder=args.temp_folder,
        output_file=args.output_file,
        book_name=args.book_name,
        start_page=args.start_page,
        end_page=args.end_page,
        number_Q_A=args.number_Q_A,
        target_information=args.target_information,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        context_length=args.context_length,
        gpu_layers=args.gpu_layers,
        max_tokens_chunk=args.max_tokens_chunk,
        arbitrary_prompt=args.arbitrary_prompt,
        model=args.model,
    )

    processor.process_book()
    processor.generate_prompts()
    processor.save_to_jsonl()


if __name__ == "__main__":
    main()
