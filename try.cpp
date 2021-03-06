#include <iostream>
#include <math.h>

const int len = 5;
using namespace std;

class poop{
public:
    poop():val1(0),val2(0.0),str1("what?"){};
    poop(int a, double b, char* c):val1(a), val2(b), str1(c){};
    ~poop(){};

    int val1;
    double val2;
    char* str1;
};

int main()
{
    /*
    poop P(1,1.2f,"who?"), *ptr, ob;
    cout << ob.val1 << ", " << ob.val2 << ", " << ob.str1;
    ptr = &P;
    cout << ptr->val1 << ", " << ptr->val2 << ", " << ptr->str1;
    cout << endl;
    */
    double out[2][2*len+1] = {0.0};

    double x = 5.0;
    double y = x;
    double gx = 0.2169;
    double gy = 0.9762;

    for (int i = -len; i <= len; i++){
        out[0][i+len] = x+i*gx;
        out[1][i+len] = y+i*gy;
    }

    int dline[2][2*len+1] = {0};
    for (int i = 0; i < 2*len+1; i++){
        dline[0][i] = (int)(out[0][i]+0.5);
        dline[1][i] = (int)(out[1][i]+0.5);
    }
    for (int i = 0; i < 2*len+1; i++){
        cout << out[0][i] << ", " << out[1][i] << endl;
    }
    cout << endl;
    for (int i = 0; i < 2*len+1; i++){
        cout << dline[0][i] << ", " << dline[1][i] << endl;
    }
    return 0;
}
