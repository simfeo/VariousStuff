using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Fake_MGP_Core.GameLogic
{
    /*
     * Base class for user.
     * Extend this if you need.
     */
    class IUser
    {
        public string name
        {
            get;
            set;
        }
        public UInt64 id
        {
            get;
            set;
        }
        public UInt64 facebookId
        {
            get;
            set;
        }

        public string SocketId
        {
            get;
            set;
        }
    }
}
