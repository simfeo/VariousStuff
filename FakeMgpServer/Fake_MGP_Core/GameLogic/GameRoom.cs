using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Fake_MGP_Core.GameLogic
{
    /*
    * This class is container for users.
    * For test purpose we need only 3 game rooms.
    */
    public class GameRoom
    {
        static List<GameRoom> rooms;
        private List<IUser> users;

        public GameRoom()
        {
            for (int i = 0; i < 5; ++i)
            {
                users.Add(new IUser());
            }
        }

        // create and init game rooms
        public static void InitRooms()
        {
            for (int i = 0; i < 3; ++i)
            {
                rooms.Add(new GameRoom());
            }
        }

    }
}
