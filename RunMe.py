import argparse
from main import q3, q4, q9, q10, qus11, qus15

def main():
    parser = argparse.ArgumentParser(description="Run coding exercises.")
    parser.add_argument('--exercise', type=str, help='Specify the exercise to run (q3, q4, q9, q10, q11, q15,q16).')
    args = parser.parse_args()

    if args.exercise:
        if args.exercise == 'q3':
            print("Question 3:")
            print()
            q3()
            print()
            print("===================")
        elif args.exercise == 'q4':
            print("Question 4:")
            print()
            q4(1)
            q4(2)
            q4(3)
            print()
            print("===================")
        elif args.exercise == 'q9':
            print("Question 9:")
            print()
            q9()
            print()
            print("===================")
        elif args.exercise == 'q10':
            print("Question 10:")
            print()
            q10()
            print("==============")
        elif args.exercise == 'q11':
            print("Question 11:")
            print()
            for lamda in [10**power for power in range(-5, 5)]:
                qus11(lamda)
            print()
            print("===================")
        elif args.exercise == 'q15':
            print("Question 15:")
            print("=========small==========")
            qus15()
            print()
            print("========Large===========")
            qus15(Large=True)
            print()
            print("===================")
        elif args.exercise == 'q16':
            print("Question 16:")
            print("========Large===========")
            qus15(Large=True)
            print()
            print("===================")
        else:
            print("Invalid exercise specified.")
    else:
        # Run all exercises
        print("Question 3:")
        print()
        q3()
        print()
        print("===================")
        print("Question 4:")
        print()
        q4(1)
        q4(2)
        q4(3)
        print()
        print("===================")
        print("Question 9:")
        print()
        q9()
        print()
        print("===================")
        print("Question 10:")
        print()
        q10()
        print("==============")
        print("Question 11:")
        print()
        for lamda in [10**power for power in range(-5, 5)]:
            qus11(lamda)
        print()
        print("===================")
        print("Question 15:")
        print("=========small==========")
        qus15()
        print()
        print("Question 16:")
        print("========Large===========")
        qus15(Large=True)
        print()
        print("===================")

if __name__ == "__main__":
    main()