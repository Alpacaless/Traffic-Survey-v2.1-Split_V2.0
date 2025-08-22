
def remove_line(fileName,lineToSkip):
    with open(fileName,'r', encoding='utf-8') as read_file:
        lines = read_file.readlines()
    currentLine = 1
    with open(fileName,'w', encoding='utf-8') as write_file:
        for line in lines:
            if currentLine == lineToSkip:
                pass
            else:
                write_file.write(line)
            currentLine += 1

