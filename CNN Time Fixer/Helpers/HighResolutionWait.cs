using System.Runtime.InteropServices;

namespace CNN_Time_Fixer.Helpers
{
    public class HighResolutionWait
    {
        [DllImport("Kernel32.dll")]
        private static extern bool QueryPerformanceCounter(out long lpPerformanceCount);

        [DllImport("Kernel32.dll")]
        private static extern bool QueryPerformanceFrequency(out long lpFrequency);

        public void Wait(int milliseconds)
        {
            QueryPerformanceFrequency(out long frequency);
            long waitTime = (frequency * milliseconds) / 1000;

            QueryPerformanceCounter(out long startTime);
            long endTime = startTime + waitTime;

            while (true)
            {
                QueryPerformanceCounter(out long currentTime);
                if (currentTime >= endTime)
                    break;
            }
        }
    }
}