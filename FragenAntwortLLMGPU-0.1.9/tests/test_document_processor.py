from FragenAntwortLLMCPU import DocumentProcessor

processor = DocumentProcessor(
    book_path="/path/to/your/book",  # Directory path without ".pdf" term
    temp_folder="/path/to/temp/folder",
    output_file="/path/to/output/QA.jsonl",
    book_name="example.pdf",
    start_page=9,
    end_page=77,
    number_Q_A="one",  # This should be a written number like "one", "two", etc.
    target_information="people, dates, agreements, organisations, companies, and locations",
    max_new_tokens=1000,
    temperature=0.1,
    context_length=2100,
    gpu_layers=400,
    max_tokens_chunk=400,
    arbitrary_prompt=""
)

processor.process_book()
processor.generate_prompts()
processor.save_to_jsonl()
