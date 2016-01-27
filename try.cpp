#include <iostream>

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

    double win_tx[9] = {0.0};
    double win_ty[9] = {0.0};
    for (int i = 0; i < 9; i++){
        cout << win_tx[i] << ", ";
    }
    cout << endl;
    for (int i = 0; i < 9; i++){
        win_tx[i] = 2.0*i;
        win_ty[i] = i/2.0;
    }
    double winxy[18];
    for(int i = 0; i < 18; i++){
        if (i/9==0){
            winxy[i] = win_tx[i];
        }
        else{
            winxy[i] = win_ty[i-9];
        }

    }

    for (int i = 0; i < 18; i++){
        cout << winxy[i] << ", ";
        if (i/9==0)
            cout << endl;
    }
    return 0;
}
