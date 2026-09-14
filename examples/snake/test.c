#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../../MLKit.h"
#include <conio.h>

#define LR 0.01f 
#define GAMMA 0.95f 
#define WIDTH 10
#define HEIGHT 10

#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define RESET   "\033[0m"

#define EPOCHS 500000

#define STATE_SIZE (WIDTH * HEIGHT + 4)

typedef struct {
    int x;
    int y;
} Point;

typedef enum {
    UP = 1,
    DOWN = -1,
    LEFT = 2,
    RIGHT = -2
} Direction;

typedef enum {
    C_APPLE,
    C_BODY,
    C_EDGE,
    C_NONE
} Collision;

typedef struct {
    int alive;
    int snake_length;
    Point snake[WIDTH * HEIGHT];
    Point head;
    Direction direction;
    Point apple;
} GameEnv;
int game_loop(GameEnv* env, RLAgent* agent, int should_log);
void delay(int ms);
void print_board(GameEnv* env);
int check_collisions(GameEnv* env);
Collision move_snake(GameEnv* env);
int delay_get_key(int ms);
void get_state(GameEnv* env, float* state);
void reset_env(GameEnv* env);



GameEnv* create_env();
int _arr_max(Layer* l) {
    int idx = 0;
    float max = 0;
    for (int i = 0; i < l->neuron_count; i++){
        if (l->outputs[i] > max) {
            max = l->outputs[i];
            idx = i;
        }
    }
    return idx;
}

int main () {
    srand(time(NULL));
    Network* model = nn_network_create(STATE_SIZE);
    nn_add_layer_dense(model, 128, activation_reLU);
    nn_add_layer_dense(model, 64, activation_reLU);
    nn_add_layer_dense(model, 16, activation_reLU);
    nn_add_layer_dense(model, 4, activation_softmax);

    nn_network_import_file(model, "./snake_brain.bin");

    RLAgent* agent = rl_agent_create(model, 1);

    

    GameEnv* env = create_env();
    game_loop(env, agent, 1);


    rl_agent_destroy(agent);
    nn_network_destroy(model);
    free(env);
}


void spawn_food(GameEnv* env) {

    while (1) {
        int is_legal = 1;

        env->apple = (Point){(int)rand() % WIDTH, (int)rand() % HEIGHT};
        for (int i = 0; i < env->snake_length; i++) {
            if (env->snake[i].x == env->apple.x && env->snake[i].y == env->apple.y) {
                is_legal = 0;
            }
        }
        if (is_legal) break;
    }

}

void eat_apple(GameEnv* env) {
    env->snake_length ++;
    
    spawn_food(env);
}

void next_head(GameEnv* env) {
    Point neck = env->snake[1];
    switch (env->direction)
    {
    case UP:
        env->head.y -= 1;
        break;
    case DOWN:
        env->head.y += 1;
        break;
    case RIGHT:
        env->head.x += 1;
        break;
    case LEFT:
        env->head.x -= 1;
        break;
    }
}


Collision move_snake(GameEnv* env) {
    Point new_head = env->head;

    switch (env->direction) {
        case UP:
            new_head.y--;
            break;

        case DOWN:
            new_head.y++;
            break;

        case LEFT:
            new_head.x--;
            break;

        case RIGHT:
            new_head.x++;
            break;
    }

    // =========================
    // WALL COLLISION
    // =========================

    if (new_head.x < 0 ||
        new_head.x >= WIDTH ||
        new_head.y < 0 ||
        new_head.y >= HEIGHT) {

        env->alive = 0;
        return C_EDGE;
    }

    // =========================
    // SAVE TAIL
    // =========================

    Point old_tail = env->snake[env->snake_length - 1];

    // =========================
    // MOVE BODY
    // =========================

    for (int i = env->snake_length - 1; i > 0; i--) {
        env->snake[i] = env->snake[i - 1];
    }

    env->head = new_head;
    env->snake[0] = new_head;

    // =========================
    // BODY COLLISION
    // =========================

    for (int i = 1; i < env->snake_length; i++) {
        if (env->snake[i].x == env->head.x &&
            env->snake[i].y == env->head.y) {

            env->alive = 0;
            return C_BODY;
        }
    }

    // =========================
    // APPLE
    // =========================

    if (env->head.x == env->apple.x &&
        env->head.y == env->apple.y) {

        if (env->snake_length < WIDTH * HEIGHT) {
            env->snake[env->snake_length] = old_tail;
            env->snake_length++;
        }

        spawn_food(env);

        return C_APPLE;
    }

    return C_NONE;
}

int game_loop(GameEnv* env, RLAgent* agent, int should_log) {
    int step = 0;
    int MAX_STEP = 60;
    int last_action_probability = 0;
    int apple_distance = abs(env->head.x - env->apple.x) + abs(env->head.y - env->apple.y);
    while (env->alive == 1) {

        if (should_log == 1) {print_board(env);printf("Score: %d hungry_for: %d action_prob: %d\n", env->snake_length-3, step, last_action_probability);}

        /*int ch = delay_get_key(200);

        switch (ch) {
            case 72:

                if (env->direction != DOWN) env->direction = UP;
                break;
            case 80:
                if (env->direction != UP) env->direction = DOWN;
                break;
            case 77:
                if (env->direction != LEFT) env->direction = RIGHT;
                break;
            case 75:
                if (env->direction != RIGHT) env->direction = LEFT;
                break;
            }*/

        float state[STATE_SIZE];

        get_state(env, state);
        nn_network_forward(agent->network, state);
        last_action_probability = (int)agent->network->output_layer->outputs[ (_arr_max(agent->network->output_layer))] * 100;
        int idx = rl_draw_action(agent->network->output_layer);
        Direction action = DOWN;
        switch (idx)
        {
        case 0:
            action = UP;
            break;
        case 1:
            action = DOWN;
            break;
        case 2:
            action = LEFT;
            break;
        case 3:
            action = RIGHT;
            break;
        }
        if (action == -1 * env->direction) {
            action = env->direction;
            
        }else {
            env->direction = action;
        }
        Collision collision_type = move_snake(env);
        if (collision_type == C_APPLE) {
            step = 0;
        }

        step += 1;
        if (should_log == 1) delay(20);
        if (collision_type == C_BODY || collision_type == C_EDGE) {
            env->alive=0;
        }



        
    }
    return env->snake_length;
}

void reset_env(GameEnv* env) {
    env->alive = 1;
    spawn_food(env);
    env->head = (Point){WIDTH/2, HEIGHT/2};
    Point head = env->head;
    env->direction = RIGHT;
    env->snake[0] = head;
    env->snake[1] = (Point){head.x-1, head.y};
    env->snake[2] = (Point){head.x-2, head.y};
    env->snake_length = 3;
}

GameEnv* create_env() {
    GameEnv* env = (GameEnv*)malloc(sizeof(GameEnv));
    env->alive = 1;
    spawn_food(env);
    env->head = (Point){WIDTH/2, HEIGHT/2};
    Point head = env->head;
    env->direction = UP;
    env->snake[0] = head;
    env->snake[1] = (Point){head.x-1, head.y};
    env->snake[2] = (Point){head.x-2, head.y};
    env->snake_length = 3;
    return env;
}

void get_state(GameEnv* env, float* state_buff) {
    for (int i = 0; i < STATE_SIZE; i++) state_buff[i] = 0;
    for (int i = 0; i < env->snake_length; i++) {
        state_buff[env->snake[i].y * WIDTH + env->snake[i].x] = 0.5;
    }
    state_buff[env->apple.y * WIDTH + env->apple.x] = -1;
    state_buff[env->head.y * WIDTH + env->head.x] = 1;
    state_buff[WIDTH*HEIGHT] = env->direction == UP;
    state_buff[WIDTH*HEIGHT+1] = env->direction == DOWN;
    state_buff[WIDTH*HEIGHT+2] = env->direction == LEFT;
    state_buff[WIDTH*HEIGHT+3] = env->direction == RIGHT;
    

}

void delay(int ms) {
    clock_t start_time = clock();
    while (clock() < start_time + ms);
}

int delay_get_key(int ms) {
    clock_t start_time = clock();
    int key = -1;
    while (clock() < start_time + ms) {
        if (kbhit()) {
            key = getch();
        }
    }
    return key;
}





int check_collisions(GameEnv* env) {
    if (env->head.x == env->apple.x && env->head.y == env->apple.y) {
        return C_APPLE; // reward +5;
    }

    for (int i = 1; i < env->snake_length; i++){
        if (env->snake[i].x == env->head.x && env->snake[i].y == env->head.y) {
            return C_BODY; // penalty -5 - body collision
        }

    }
    if (env->head.x < 0 || env->head.x >= WIDTH || env->head.y < 0 || env->head.y >= HEIGHT) {
        env->alive = 0;
        return C_EDGE; 
    }
    return C_NONE;
}



void print_board(GameEnv* env) {
    printf("\033[H\033[J");
    for (int i = 0; i < WIDTH+1; i++) printf("# ");
    printf("\n");
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = -1; x < WIDTH+1; x++) {
            if (x == -1 || x == WIDTH) {
                printf(BLUE "# ");
                continue;
            }
            char c = ' ';

            for (int i = 0; i < env->snake_length; i++) {
                if (env->snake[i].x == x && env->snake[i].y == y) {
                    c = 'o';
                }
            }
            if (env->head.x == x && env->head.y == y) {
                c = 'O';
            };
            if (env->apple.x == x && env->apple.y == y) {
                c = '@';
            }
            printf(RESET);
            printf(c == '@' ? RED : c == 'O' || c== 'o' ? GREEN : RESET); 
            printf( "%c ", c);
        
        }
        printf("\n");
    }
    for (int i = 0; i < WIDTH; i++) printf("# ");
    printf("\n" RESET);


}


