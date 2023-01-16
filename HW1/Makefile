CPPFLAGS = -std=c++17 -Wall -Werror -pedantic -ggdb -pthread

hw1_setup : hw1_setup.cpp
	g++ $(CPPFLAGS) $< -o $@

example : example.cpp ThreadGroup.h
	g++ $(CPPFLAGS) $< -o $@

hw1 : hw1.cpp ThreadGroup.h
	g++ $(CPPFLAGS) $< -o $@

all : example hw1_setup hw1
	@echo "Made it all!"

clean :
	rm -f hw1_setup example hw1