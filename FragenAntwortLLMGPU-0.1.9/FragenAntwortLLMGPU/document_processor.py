import fitz  # PyMuPDF for PDF processing
from tokenizers import Tokenizer  # HuggingFace tokenizers
from semantic_text_splitter import TextSplitter  # Text splitting for large documents
from langchain.llms import CTransformers  # Interface for large language models
from langchain.prompts import PromptTemplate  # Template for prompts
import os  # Operating system interface
from langchain.chains import LLMChain  # Chain for linking LLM and prompt
import gc  # Garbage collector interface
import torch  # PyTorch for tensor computations
import json  # JSON for data interchange format

class DocumentProcessor:
    def __init__(self, book_path, temp_folder, output_file, book_name, start_page, end_page, number_Q_A, target_information, max_new_tokens=500, temperature=0.1, context_length=1000, gpu_layers=100, max_tokens_chunk=400, arbitrary_prompt=""):
        """
        Initialize the DocumentProcessor with the given parameters.

        Parameters:
        book_path (str): Path to the directory containing the book PDF files.
        temp_folder (str): Directory to store temporary files.
        output_file (str): Path to the output JSONL file.
        book_name (str): Name of the book PDF file.
        start_page (int): Start page number for processing.
        end_page (int): End page number for processing.
        number_Q_A (str): Number of questions and answers to generate (as a written number).
        target_information (str): Focus of the questions and answers.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature for the LLM.
        context_length (int): Maximum context length for the LLM.
        gpu_layers (int): Number of GPU layers to use.
        max_tokens_chunk (int): Maximum tokens per text chunk.
        arbitrary_prompt (str): Custom prompt for generating questions and answers.
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

        # Default prompt if arbitrary_prompt is not provided
        if len(self.arbitrary_prompt) > 10:
            self.question_p = self.arbitrary_prompt
        else:
            self.question_p = (
                f"I need you to extract up to {self.number_Q_A} sets of complex questions and their corresponding answers from the provided text. The questions and answers should focus on {self.target_information} and must be based directly on the input text. Please ensure the questions are meaningful and avoid asking about the main idea or purpose of the text. Do not use pronouns or phrases like 'this period' and 'in this text' in your questions. Complex questions should require answers that involve two or more steps. Use specific names and terms for people, locations, agreements, dates, events, and {self.target_information} instead of pronouns such as 'he', 'she', 'they', 'him', 'her', 'them', 'this', 'these', or 'those'. For example, ask 'What is the nature of 17th-century society?' instead of 'What is the nature of this society?' Each question must start with a number (e.g., '1. What is ...'). Provide each question immediately followed by its answer."
            )

        # Mapping written numbers to their numeric equivalents
        self.written_numbers = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twenty-one"
        ]
        self.number_dict = {self.written_numbers[i]: i + 1 for i in range(len(self.written_numbers))}
        
        # Valid numbers for generating Q&A sets
        self.valid_number = [index for index in range(1, self.number_dict[self.number_Q_A] + 1)]
        
        # Configuration for the LLM
        self.config = {
            'max_new_tokens': self.max_new_tokens, 
            'temperature': self.temperature, 
            'context_length': self.context_length, 
            'gpu_layers': self.gpu_layers
        }
        
        # Prompt template for LLM
        self.template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly from the context.
        Answer the question below from context below :
        {context}
        {question} [/INST] </s>"""
        
        # Initialize tokenizer and text splitter
        self.tokenizer22 = Tokenizer.from_pretrained("bert-base-uncased")
        self.splitter = TextSplitter.from_huggingface_tokenizer(self.tokenizer22, self.max_tokens_chunk)
        
        # List to store all prompts
        self.all_prompts = []

    def process_book(self):
        """
        Process the book to generate Q&A pairs.

        This method extracts text from the specified pages of the book, splits the text into chunks,
        and generates Q&A pairs using the LLM.
        """
        key = list(self.books.keys())[0]
        with open(os.path.join(self.temp_folder, key + "_QA.txt"), "w", encoding="utf-8") as prompts_write:
            path = os.path.join(self.book_path, key)
            count = 0
            with fitz.open(path) as doc:
                whole_document = ""
                # Extract text from specified pages
                for page in doc:
                    if count >= self.books[key][0] and count < self.books[key][1]:
                        text = page.get_text("blocks")
                        for e in text:
                            whole_document += e[-3].replace("¬\n", "").replace("¬ \n", "")
                    count += 1
                # Split text into chunks
                chunks = self.splitter.chunks(whole_document.strip())
            
            print("chunks: ", len(chunks))
            
            # Initialize LLM
            llm = CTransformers(
                model='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',
                model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                config=self.config,
                threads=os.cpu_count()
            )
            
            count = 0
            for context in chunks:
                prompt = PromptTemplate(template=self.template, input_variables=["question", "context"])
                llm_chain = LLMChain(prompt=prompt, llm=llm)
                response = llm_chain.run({"question": self.question_p, "context": context})
                # Write Q&A pairs to file
                prompts_write.write("book_name: " + key + " Chunk: " + context.replace("\n", " ") + " \n " + response)
                count += 1
                prompts_write.flush()
                prompts_write.write("\n******************************\n")
                print("Chunk ",str(count)," ***done** ")
                print("book_name: " + key + " \t Chunk: " + context.replace("\n", " ") + " \n " + response)
                print("Chunk ",str(count)," ***done** ")
                print("*******************************")
                
            # Clean up to free memory
            print("'process_book =_= done '")
            try:
                del llm
            except:
                pass
            try:
                del response
            except:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    def generate_prompts(self):
        """
        Generate prompts from the processed Q&A pairs.

        This method reads the generated Q&A pairs, formats them, and stores them in the all_prompts list.
        """
        Chunks = []
        Begin = '<s> ### Question: { '
        key = list(self.books.keys())[0]
        with open(os.path.join(self.temp_folder, key + "_QA.txt"), "r", encoding="utf-8") as reader:
            flag = True
            Allow = False
            new_line = ""
            for line in reader:
                if "****" not in line and line.rstrip().lstrip().strip() != "\n":
                    if flag:
                        book_name1 = " the books' name is " + line.split("Chunk:")[0].split("book_name:")[1].replace(".pdf", "").rstrip().lstrip().strip()
                        Chunks.append(line.split("Chunk:")[1])
                        flag = False
                    if "?" in line:
                        for tt in self.valid_number:
                            if str(tt) + ". " in line:
                                Allow = True
                                break
                            else:
                                Allow = False
                        if Allow:
                            if new_line:  # If there is already a new_line, append it to all_prompts before starting a new one
                                self.all_prompts.append(new_line.rstrip().lstrip().strip().replace("\n", "") + "} </s> \n \n ")
                            new_line = Begin + line.replace("\n", "").replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "")\
                            .replace("5.", "").replace("6.", "").replace("7.", "").replace("8.", "")\
                            .replace("9.", "").replace("10.", "").replace("11.", "").replace("12.", "") + book_name1 + " } ### Answer: { "
                    else:
                        if Allow:
                            new_line = new_line + line.replace("Answer:", "")
                else:
                    if line.rstrip().lstrip().strip() != "\n":
                        if new_line:
                            self.all_prompts.append(new_line.rstrip().lstrip().strip().replace("\n", "") + "} </s> \n \n ")
                        Allow = False    
                        flag = True
                        new_line = ""

        # Ensure the last prompt is added if there is any
        if new_line:
            self.all_prompts.append(new_line.rstrip().lstrip().strip().replace("\n", "") + "} </s> \n \n ")

        print("*************************************")
        for each_prompt in self.all_prompts:
            print(each_prompt)
            print("*************************************")
        print("*** done ***")

    def save_to_jsonl(self):
        """
        Save the generated prompts to a JSONL file.

        This method writes the formatted prompts to the specified output JSONL file.
        """
        with open(self.output_file, "w", encoding="latin1") as output_jsonl_file:
            for item in self.all_prompts:
                json_object = {"text": item}
                output_jsonl_file.write(json.dumps(json_object) + "\n")

# Example usage
if __name__ == "__main__":
    book_path = "/home/devuser/Books"  # Directory path without ".pdf" term
    temp_folder = "/home/devuser/Test_temp"
    output_file = "/home/devuser/QA.jsonl"
    book_name = "/home/devuser/example.pdf"
    start_page = 9
    end_page = 77
    number_Q_A = "one"  # This should be a written number like "one", "two", etc.
    target_information = "people, dates, agreements, organisations, companies and locations"
    max_new_tokens = 1000
    temperature = 0.1
    context_length = 2100
    #gpu_layers = 400
    max_tokens_chunk = 400
    arbitrary_prompt = ""

    print("started")
    processor = DocumentProcessor(book_path, temp_folder, output_file, book_name, start_page, end_page, number_Q_A, target_information, max_new_tokens, temperature, context_length, gpu_layers, max_tokens_chunk, arbitrary_prompt)
    print("process_book")
    processor.process_book()
    processor.generate_prompts()
    processor.save_to_jsonl()
