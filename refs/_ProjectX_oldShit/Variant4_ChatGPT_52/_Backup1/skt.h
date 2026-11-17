#ifndef SKT_H
#define SKT_H

#define SKT_MAGIC 0x534B5401UL /* 'SKT'+v1 */

typedef struct {
    unsigned long magic;
    unsigned short inputs;
    unsigned short hidden;
    unsigned short outputs;
    unsigned long samples;
} SKTHeader;

/* layout:
   [SKTHeader]
   [float w1[inputs][hidden]]
   [float w2[hidden][outputs]]
*/

#endif
