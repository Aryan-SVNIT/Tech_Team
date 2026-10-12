def take_text_input():
    print("Paste your forwarded message. Type END on a new line to finish.\n")
    lines = []
    while True:
        try: line = input()
        except EOFError: break
        if line.strip().upper() == "END": break
        lines.append(line)
    text = "\n".join(lines).strip()
    return text or None

if __name__ == "__main__":
    text = take_text_input()
    if text: print("\n--- Extracted Text ---\n" + text)
