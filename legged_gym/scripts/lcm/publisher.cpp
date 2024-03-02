#include <stdio.h>
#include <lcm/lcm-cpp.hpp>
#include "action_t.hpp"

lcm::LCM lc; //"udpm://224.0.55.55:5001?ttl=225"

class Handler
{
    public:
        ~Handler() {}
        
        void handleMessage(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan,
                const action_t*msg)
        {
                printf("Received message on channel \"%s\":\n");


                action_t actionspace ;
        }
};

int main(int arg, char ** argv)
{
    

    if(!lc.good())
        return 1;

    Handler handlerObject;
    lc.subscribe("ACTION",&Handler::handleMessage, &handlerObject);
    
    
    while(0 == lc.handle());

    return 0;
}
