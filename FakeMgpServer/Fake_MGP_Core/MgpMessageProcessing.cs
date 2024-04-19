using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Fake_MGP_Core
{
    public class MgpMessageProcessing
    {
        public static Dictionary<string, MgpMessageProcessing> websocketsDict;

        private WebSocket webSocket;
        private WebSocketReceiveResult websocketRecieve;

        public MgpMessageProcessing(WebSocket webSocket, WebSocketReceiveResult result)
        {
            this.webSocket = webSocket;
            this.websocketRecieve = result;
        }

        public async Task Send( ArraySegment<byte> buffer)
        {
            await webSocket.SendAsync(buffer, websocketRecieve.MessageType, websocketRecieve.EndOfMessage, CancellationToken.None);
        }

        public Task<WebSocketReceiveResult> Receive(ArraySegment<byte> buffer)
        {
            var str = System.Text.Encoding.Default.GetString(buffer);
            try
            {
                var ss = JObject.Parse(str);
                JObject jval = JObject.Parse(@"{""command"":true}");
                Task.Run(()=> Send(Encoding.ASCII.GetBytes(jval.ToString())));
            }
            catch
            {
                //not json
            }
            return webSocket.ReceiveAsync(buffer, CancellationToken.None);
        }
    }
}
