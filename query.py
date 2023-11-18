def boolean_query(query: str):
    # TODO: boolean retrieval, AND only
    terms = query.split()

    return []



if __name__ == '__main__':
    print("Search Engine Start.")

    while True:
        user_input = input("Enter a query (or 'exit' to quit): ")

        if user_input.lower() == 'exit':
            print("Exiting the Search Engine. Goodbye!")
            break

        results = boolean_query(user_input)
        # TODO: show results