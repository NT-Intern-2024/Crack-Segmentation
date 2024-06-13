def print_to_file(file_path: str, text: str):
    with open(file_path, "w") as output_file:
        print(text, file=output_file)