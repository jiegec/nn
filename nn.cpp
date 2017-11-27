#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fstream>

typedef float fp;
const int MAX_EDGE = 20;
const int TYPE_INPUT = 0;
const int TYPE_HIDDEN = 1;
const int TYPE_OUTPUT = 2;
const fp learning_rate = 0.5;
const int output_num = 3;
const int hidden_num = 3;
const int hidden_layers = 1;
const int input_num = 2;

int i[input_num], h[hidden_num * hidden_layers], o[output_num];

// output, gradient
using unit = std::pair<fp, fp>;

class Node {
public:
    int type;
    int from[MAX_EDGE];
    int edge_from;
    int to[MAX_EDGE];
    int edge_to;
    unit utop;
};

class SigmoidNode : public Node {
public:
    unit weight[MAX_EDGE];
    unit bias;

    void forward();

    void backward();
};


SigmoidNode pool[100];
int top = 0;

fp sigmoid(fp input) {
    return 1 / (1 + exp(-input));
}


void SigmoidNode::forward() {
    fp net = 0;
    for (int j = 0; j < edge_from; j++) {
        net += pool[from[j]].utop.first * weight[j].first;
        weight[j].second = 0.0;
    }
    net = net + bias.first;
    bias.second = 0.0;
    utop = unit(sigmoid(net), 0.0);
}

void SigmoidNode::backward() {
    for (int j = 0; j < edge_from; j++) {
        weight[j].second += pool[from[j]].utop.first * utop.first * (1 - utop.first) * utop.second;
        pool[from[j]].utop.second += weight[j].first * utop.first * (1 - utop.first) * utop.second;
    }
    bias.second += utop.first * (1 - utop.first) * utop.second;
}

fp sqr(fp input) {
    return input * input;
}

int create_node(int type) {
    SigmoidNode &new_node = pool[top++];
    new_node.type = type;
    new_node.bias = unit((fp) rand() / RAND_MAX, 0.0);
    new_node.edge_from = new_node.edge_to = 0;
    return top - 1;
}

void add_edge(int from, int to) {
    SigmoidNode &edge1 = pool[from];
    SigmoidNode &edge2 = pool[to];
    edge1.to[edge1.edge_to++] = to;
    edge2.from[edge2.edge_from++] = from;
    edge2.weight[edge2.edge_from - 1] = unit((fp) rand() / RAND_MAX, 0.0);
}

void update_value() {
    for (int i = 0; i < top; i++) {
        SigmoidNode &cur = pool[i];
        if (cur.type != TYPE_INPUT) {
            cur.forward();
        }
    }
}

void create_nodes(int num, int type, int array[]) {
    for (int i = 0; i < num; i++) {
        array[i] = create_node(type);
    }
}

void print() {
    std::ofstream out;
    out.open("output.dot", std::ios::out);
    out << "graph {" << std::endl;
    for (int i = 0; i < top; i++) {
        if (pool[i].edge_from) {
            for (int j = 0; j < pool[i].edge_from; j++) {
                out << pool[i].from[j] << " -- " << i \
 << " [label=\"" \
 << "weight=" << pool[i].weight[j].first \
 << "\"]"                          \
 << std::endl;
            }
        }
        out << i << " [label=\"type: " << pool[i].type << " w: " << pool[i].utop.first << "\"]" << std::endl;
    }
    out << "}" << std::endl;
}

void train(const fp input[], fp target[]) {
    for (int ii = 0; ii < input_num; ii++)
        pool[i[ii]].utop.first = input[ii];
    for (int it = 0; it < 1000; it++) {
        update_value();
        fp total_error = 0;
        for (int i = 0; i < output_num; i++) {
            total_error += sqr(target[i] - pool[o[i]].utop.first);
        }
        total_error /= 2;
        // printf("it#%d error: %f\n", it, total_error);
        if (total_error < 1e-6)
            break;
        for (int oo = 0; oo < output_num; oo++)
            pool[o[oo]].utop.second = -(target[oo] - pool[o[oo]].utop.first) / 2;

        for (int i = top - 1; i >= 0; i--) {
            if (pool[i].type != TYPE_INPUT) {
                pool[i].backward();
            }
        }
        for (int i = top - 1; i >= 0; i--) {
            if (pool[i].type != TYPE_INPUT) {
                for (int j = 0; j < pool[i].edge_from; j++) {
                    pool[i].weight[j].first -= learning_rate * pool[i].weight[j].second;
                }
                pool[i].bias.first -= learning_rate * pool[i].bias.second;
            }
        }
        /*
        for (int i = 0; i < top; i++) {
            pool[i].delta_weight = 0;
        }
        // BackPropagation
        for (int i = 0; i < output_num; i++) {
            int id = o[i];
            pool[id].delta_weight = pool[id].out - target[i];
        }

        for (int i = top; i >= 0; i--) {
            if (pool[i].type != TYPE_INPUT) {
                pool[i].delta_weight *= pool[i].out * (1 - pool[i].out);
                for (int j = 0; j < pool[i].edge_from; j++) {
                    int edge_from = pool[i].from[j];
                    pool[edge_from].delta_weight += pool[i].delta_weight * pool[i].weight[j];
                    pool[i].weight[j] -= learning_rate * pool[i].delta_weight * pool[edge_from].out;
                }
            }
        }
         */
    }
}

int main() {
    srand(time(0));
    create_nodes(input_num, TYPE_INPUT, i);
    create_nodes(hidden_num * hidden_layers, TYPE_HIDDEN, h);
    create_nodes(output_num, TYPE_OUTPUT, o);

    for (int layer = 0; layer < hidden_layers; layer++) {
        for (int ii = 0; ii < hidden_num; ii++) {
            int cur = layer * hidden_num + ii;
            if (layer == 0) {
                for (int j = 0; j < input_num; j++) {
                    add_edge(i[j], h[cur]);
                }
            } else {
                for (int j = 0; j < hidden_num; j++) {
                    add_edge(h[(layer - 1) * hidden_num + j], h[cur]);
                }
            }

            if (layer == hidden_layers - 1) {
                for (int j = 0; j < output_num; j++) {
                    add_edge(h[cur], o[j]);
                }
            }
        }
    }

    fp input[input_num];
    fp output[output_num];
    for (int c = 0; c < 100; c++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                input[0] = (fp) i;
                input[1] = (fp) j;
                output[0] = (fp) (i ^ j);
                output[1] = (fp) (i & j);
                output[2] = (fp) (i | j);
                train(input, output);
            }
        }
    }

    printf("Let's test!\n");
    for (;;) {
        int a, b;
        if (scanf("%d%d", &a, &b) != 2)
            break;
        pool[i[0]].utop.first = a;
        pool[i[1]].utop.first = b;
        update_value();
        printf("answer: a^b=%d a&b=%d a|b=%d\n", pool[o[0]].utop.first > 0.5, pool[o[1]].utop.first > 0.5,
               pool[o[2]].utop.first > 0.5);
        input[0] = a, input[1] = b, output[0] = a ^ b, output[1] = a & b, output[2] = a | b;
        // train(input, output);
        //   pool[i[0]].out = a;
        //   pool[i[1]].out = b;
        //   update_value();
        //   printf("after-training answer: %f %f\n", pool[o[0]].out, pool[o[1]].out);
    }

    print();
    return 0;
}
