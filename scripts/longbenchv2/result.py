import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, default='./results/longbenchv2')
    parser.add_argument('--output_file', type=str, default='./results/longbenchv2/result.txt')
    parser.add_argument('--compensated', action='store_true')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file
    compensated = args.compensated

    if not os.path.exists(input_dir):
        print(f"Cannot find '{input_dir}'")
        return

    files = os.listdir(input_dir)
    output = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong"]

    for file in files:
        filename = os.path.join(input_dir, file)
        
        if os.path.isdir(filename) or filename == output_file:
            continue
            
        print(f"{filename}")
        try:
            with open(filename, encoding='utf-8') as f:
                pred_data = json.load(f)
        except Exception:
            try:
                with open(filename, encoding='utf-8') as f:
                    pred_data = [json.loads(line) for line in f]
            except Exception:
                continue
                
        if not pred_data:
            continue

        easy, hard, short, medium, long = 0, 0, 0, 0, 0
        easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
        
        for pred in pred_data:
            acc = int(pred.get('judge', 0))
            if compensated and pred.get("pred") is None:
                acc = 0.25
                
            if pred.get("difficulty") == "easy":
                easy += 1
                easy_acc += acc
            else:
                hard += 1
                hard_acc += acc

            if pred.get('length') == "short":
                short += 1
                short_acc += acc
            elif pred.get('length') == "medium":
                medium += 1
                medium_acc += acc
            else:
                long += 1
                long_acc += acc

        print(easy_acc, hard_acc, short_acc, medium_acc, long_acc)
        name = '.'.join(file.split('.')[:-1])
        
        overall_len = len(pred_data)
        overall_score = str(round(100*(easy_acc+hard_acc)/overall_len, 1)) if overall_len else "0.0"
        easy_score = str(round(100*easy_acc/easy, 1)) if easy else "0.0"
        hard_score = str(round(100*hard_acc/hard, 1)) if hard else "0.0"
        short_score = str(round(100*short_acc/short, 1)) if short else "0.0"
        medium_score = str(round(100*medium_acc/medium, 1)) if medium else "0.0"
        long_score = str(round(100*long_acc/long, 1)) if long else "0.0"

        output.append(f"{name}\t{overall_score}\t{easy_score}\t{hard_score}\t{short_score}\t{medium_score}\t{long_score}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
        

if __name__ == '__main__':
    main()