"""Hand-authored pilot examples, not recorded user conversations.

Keep held-out prompts out of synthetic-data generation. These small splits are a
plumbing pilot, not sufficient evidence for deployment or a production dataset.
"""


def call(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": name, "arguments": arguments}}
    ]}


def reply(text):
    return {"role": "assistant", "content": text}


TRAIN = [
    ("What's the weather today?", call("get_weather", day="today")),
    ("Will it rain tomorrow?", call("get_weather", day="tomorrow")),
    ("Check the forecast for Denver tomorrow", call("get_weather", location="Denver", day="tomorrow")),
    ("Weather in Rome on October 12", call("get_weather", location="Rome", day="10-12")),
    ("What time is it?", call("get_current_datetime", part="time")),
    ("Tell me today's date", call("get_current_datetime", part="date")),
    ("Give me the date and time", call("get_current_datetime", part="both")),
    ("What are my weather settings?", call("get_weather_preferences")),
    ("Which city did I save?", call("get_weather_preferences")),
    ("Set my default city to Denver", call("set_weather_preferences", default_city="Denver")),
    ("Use Celsius by default", call("set_weather_preferences", temperature_unit="celsius")),
    ("Switch my default temperature units to Fahrenheit", call("set_weather_preferences", temperature_unit="fahrenheit")),
    ("Clear my default city", call("set_weather_preferences", default_city="")),
    ("Set a timer for ten minutes", call("create_timer", seconds=600)),
    ("Give me five minutes for the pasta", call("create_timer", seconds=300, label="pasta")),
    ("Start a timer for one hour and fifteen minutes", call("create_timer", seconds=4500)),
    ("Set a thirty second timer", call("create_timer", seconds=30)),
    ("List my timers", call("list_timers")),
    ("How much time is left on my timers?", call("list_timers")),
    ("Cancel timer number one", call("cancel_timer", id=1)),
    ("Stop timer twenty two", call("cancel_timer", id=22)),
    ("Cancel all my timers", call("cancel_all_timers")),
    ("Set an alarm for tomorrow at seven AM", call("create_alarm", hour=7, minute=0, tomorrow=True)),
    ("Wake me at 6:30 AM every weekday", call("create_alarm", hour=6, minute=30, recurrence="weekdays")),
    ("Set a daily alarm for 8 PM", call("create_alarm", hour=20, minute=0, recurrence="daily")),
    ("Set an alarm for 19:45", call("create_alarm", hour=19, minute=45)),
    ("Show my alarms", call("list_alarms")),
    ("Which alarms do I have set?", call("list_alarms")),
    ("Cancel alarm number two", call("cancel_alarm", id=2)),
    ("Delete alarm twelve", call("cancel_alarm", id=12)),
    ("Remove all alarms", call("cancel_all_alarms")),
    ("Hello", reply("Hi! What can I help with?")),
    ("How are you?", reply("I'm here and ready to help. How are you doing?")),
    ("What is a black hole?", reply("A black hole is a region where gravity is so strong that even light cannot escape.")),
    ("Explain how rain forms", reply("Water vapor condenses into droplets in clouds. When droplets become heavy enough, they fall as rain.")),
    ("Why do alarms help people wake up?", reply("Their sound draws your attention and can wake you at a planned time.")),
    ("Tell me a joke", reply("Why did the bicycle fall over? It was two-tired.")),
    ("What's nine times six?", reply("Fifty-four.")),
    ("Thanks", reply("You're welcome.")),
    ("Set a timer", reply("How long should the timer run?")),
    ("Wake me at seven", reply("Do you mean seven AM or seven PM?")),
    ("Cancel an alarm", reply("Which alarm ID should I cancel?")),
    ("Book me a taxi", reply("I can't book taxis yet.")),
    ("Send a WhatsApp message to Alex", reply("WhatsApp messaging isn't available yet.")),
    ("Write down that I need milk", reply("I can't save notes yet.")),
    ("Play some music", reply("I can't play music yet.")),
    ("Don't change my default city", reply("Okay, I'll leave it as it is.")),
    ("Set an alarm for next Friday at 8 AM", reply("I currently support alarms for the next occurrence, tomorrow, daily, or weekdays. Which would you like?")),
]

VALIDATION = [
    ("Forecast for Lisbon tomorrow please", call("get_weather", location="Lisbon", day="tomorrow")),
    ("Tell me the current time", call("get_current_datetime", part="time")),
    ("What date is today?", call("get_current_datetime", part="date")),
    ("Read my saved preferences", call("get_weather_preferences")),
    ("Make Lisbon my default city", call("set_weather_preferences", default_city="Lisbon")),
    ("Start a twelve minute timer", call("create_timer", seconds=720)),
    ("Show active timers", call("list_timers")),
    ("Delete timer number six", call("cancel_timer", id=6)),
    ("Stop every timer", call("cancel_all_timers")),
    ("Alarm at 21:15 please", call("create_alarm", hour=21, minute=15)),
    ("List active alarms", call("list_alarms")),
    ("Remove alarm six", call("cancel_alarm", id=6)),
    ("Cancel every alarm", call("cancel_all_alarms")),
    ("How do clouds form?", reply("Clouds form when water vapor cools and condenses into tiny droplets or ice crystals.")),
    ("Set an alarm for nine", reply("Nine AM or nine PM?")),
    ("Order a pizza", reply("I can't place food orders yet.")),
]

TEST = [
    ("Do I need an umbrella tomorrow?", call("get_weather", day="tomorrow")),
    ("What's the weather like in Seattle?", call("get_weather", location="Seattle")),
    ("Weather on September 28th", call("get_weather", day="09-28")),
    ("Forecast for Boston on September 28, 2026", call("get_weather", location="Boston", day="2026-09-28")),
    ("Could you check the time for me?", call("get_current_datetime", part="time")),
    ("I need today's date please", call("get_current_datetime", part="date")),
    ("What's today's date and the time right now?", call("get_current_datetime", part="both")),
    ("What is my default city?", call("get_weather_preferences")),
    ("Save Boston as my default location", call("set_weather_preferences", default_city="Boston")),
    ("From now on give temperatures in Celsius", call("set_weather_preferences", temperature_unit="celsius")),
    ("Give me eight minutes", call("create_timer", seconds=480)),
    ("Start a tea timer for three minutes", call("create_timer", seconds=180, label="tea")),
    ("Timer for two hours and thirty minutes", call("create_timer", seconds=9000)),
    ("What timers are still running?", call("list_timers")),
    ("Please cancel timer number twenty one", call("cancel_timer", id=21)),
    ("Stop all the running timers", call("cancel_all_timers")),
    ("Wake me tomorrow at 8:15 AM", call("create_alarm", hour=8, minute=15, tomorrow=True)),
    ("Set a weekday alarm at 9 AM", call("create_alarm", hour=9, minute=0, recurrence="weekdays")),
    ("Alarm every day at 10 PM", call("create_alarm", hour=22, minute=0, recurrence="daily")),
    ("What alarms are active?", call("list_alarms")),
    ("Cancel alarm number one", call("cancel_alarm", id=1)),
    ("Delete all the alarms", call("cancel_all_alarms")),
    ("How do black holes form?", reply("Some form when massive stars collapse under their own gravity at the end of their lives.")),
    ("How's it going?", reply("I'm ready to help. How's your day going?")),
    ("Explain how a timer works", reply("A timer counts down a duration and signals when that duration ends.")),
    ("Tell me a story about the weather", reply("A little cloud drifted over a thirsty garden and decided to stay for a shower.")),
    ("I want an alarm for eight", reply("Do you mean eight AM or eight PM?")),
    ("Cancel my timer", reply("Which timer ID should I cancel?")),
    ("Please book a cab", reply("I can't book a cab yet.")),
    ("Text my brother that I'm late", reply("I can't send messages yet.")),
    ("Turn off the bedroom lights", reply("I can't control lights yet.")),
    ("Remember to buy eggs", reply("I can't save notes or reminders yet.")),
]


def rows(split):
    pairs = {"train": TRAIN, "validation": VALIDATION, "test": TEST}[split]
    result = [{"id": f"{split}-{i:03d}", "group": f"{split}-{i:03d}",
               "messages": [{"role": "user", "content": prompt}], "target": target,
               "category": "tool" if target.get("tool_calls") else "text"}
              for i, (prompt, target) in enumerate(pairs)]
    # Explicitly scripted histories: evaluate continuation independently of earlier predictions.
    city = {"train": "Denver", "validation": "Lisbon", "test": "Seattle"}[split]
    result.append({"id": f"{split}-followup", "group": f"{split}-followup", "category": "tool",
                   "messages": [{"role": "user", "content": f"Weather in {city} today"},
                                {"role": "assistant", "content": f"It's clear in {city} today."},
                                {"role": "user", "content": "And tomorrow?"}],
                   "target": call("get_weather", location=city, day="tomorrow")})
    result.append({"id": f"{split}-disabled", "group": f"{split}-disabled", "category": "text",
                   "tools": [], "messages": [{"role": "user", "content": pairs[0][0]}],
                   "target": reply("Weather lookup isn't available in this session.")})
    result.append({"id": f"{split}-clarification", "group": f"{split}-clarification", "category": "tool",
                   "messages": [{"role": "user", "content": "Set a timer"},
                                {"role": "assistant", "content": "How long should it run?"},
                                {"role": "user", "content": {"train": "Six minutes", "validation": "Eleven minutes", "test": "Four minutes"}[split]}],
                   "target": call("create_timer", seconds={"train": 360, "validation": 660, "test": 240}[split])})
    return result
