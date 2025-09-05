# ABOUTME: This script allows running a single scenario for testing and analysis
# ABOUTME: Useful for debugging strategy performance on specific scenarios

from berghain_bot import BerghainBot
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_single_scenario.py <scenario_number>")
        print("Where scenario_number is 1, 2, or 3")
        sys.exit(1)
    
    try:
        scenario = int(sys.argv[1])
        if scenario not in [1, 2, 3]:
            raise ValueError("Scenario must be 1, 2, or 3")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    bot = BerghainBot()
    
    print(f"Playing Scenario {scenario}...")
    print("-" * 40)
    
    try:
        result = bot.play_game(scenario)
        
        print(f"\nFinal Result:")
        print(f"Status: {result['status']}")
        print(f"Rejections: {result['rejected_count']}")
        print(f"Admissions: {result['admitted_count']}")
        
        if result['status'] == 'completed':
            print(f"\n🏆 SUCCESS! You minimized rejections to {result['rejected_count']}")
        else:
            print(f"\n💀 Game failed with {result['rejected_count']} rejections")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()