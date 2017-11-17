

# Initial Permutation Matrix
IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]

# Inverse Permutation Matrix
InvP = [40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25]

#Permutation made after each SBox substitution
P = [16, 7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26, 5, 18, 31, 10,
     2, 8, 24, 14, 32, 27, 3, 9,
     19, 13, 30, 6, 22, 11, 4, 25]

# Initial permutation on key
PC_1 = [57, 49, 41, 33, 25, 17, 9,
        1, 58, 50, 42, 34, 26, 18,
        10, 2, 59, 51, 43, 35, 27,
        19, 11, 3, 60, 52, 44, 36,
        63, 55, 47, 39, 31, 23, 15,
        7, 62, 54, 46, 38, 30, 22,
        14, 6, 61, 53, 45, 37, 29,
        21, 13, 5, 28, 20, 12, 4]

# Permutation applied after shifting key (i.e gets Ki+1)
PC_2 = [14, 17, 11, 24, 1, 5, 3, 28,
        15, 6, 21, 10, 23, 19, 12, 4,
        26, 8, 16, 7, 27, 20, 13, 2,
        41, 52, 31, 37, 47, 55, 30, 40,
        51, 45, 33, 48, 44, 49, 39, 56,
        34, 53, 46, 42, 50, 36, 29, 32]

# Expand matrix to obtain 48bit matrix
E = [32, 1, 2, 3, 4, 5,
     4, 5, 6, 7, 8, 9,
     8, 9, 10, 11, 12, 13,
     12, 13, 14, 15, 16, 17,
     16, 17, 18, 19, 20, 21,
     20, 21, 22, 23, 24, 25,
     24, 25, 26, 27, 28, 29,
     28, 29, 30, 31, 32, 1]

# SBOX represented as a three dimentional matrix
# --> SBOX[block][row][column]
SBOX = [        
[
 [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
 [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
 [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
 [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
],
[
 [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
 [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
 [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
 [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
],
[
 [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
 [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
 [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
 [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
],
[
 [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
 [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
 [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
 [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
],  
[
 [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
 [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
 [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
 [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
], 
[
 [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
 [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
 [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
 [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
], 

[[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
 [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
 [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
 [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
],
[
 [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
 [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
 [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
 [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
]
]

# Shift Matrix for each round of keys
SHIFT = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

def str_to_bitarray(s):
    # Converts string to a bit array.
    bitArr = list()
    for byte in s:
        bits = bin(byte)[2:] if isinstance(byte, int) else bin(ord(byte))[2:]
        while len(bits) < 8:
            bits = "0"+bits  # Add additional 0's as needed
        for bit in bits:
            bitArr.append(int(bit))
    return bitArr

def bitarray_to_str(bitArr):
    # Converts bit array to string
    result = ''
    for i in range(0,len(bitArr),8):
        byte = bitArr[i:i+8]
        s = ''.join([str(b) for b in byte])
        result = result+chr(int(s,2))
    return result

class DES():
    def __init__(self, plaintext, password):
        # Converting the given strings and store them in a list
        keyBit = str_to_bitarray(password)
        textBit = str_to_bitarray(plaintext)
        self.password = keyBit
        self.plaintext = textBit
        self.keylist = list()
        self.createKeys()

    def left_shift(self, a, round_num):
        # Shifts a list based on a round number
        num_shift = SHIFT[round_num]
        ### YOUR CODE HERE ###
        keyPostShift = [a[num_shift:] + a[:num_shift]]
        keyPostShift = [ x for y in keyPostShift for x in y]
        return keyPostShift

    def createKeys(self):
        # This functions creates the keys and stores them in keylist.
        # These keys should be generated using the password.
        ### YOUR CODE HERE ###
        initialKeyArray = self.permute(self.password, PC_1)     #calling the permute function
        leftKeyArray = initialKeyArray[:28]
        rightKeyArray = initialKeyArray[28:]                    #splitting the permuted key to two halves
        i = 0
        for i in range(0,16):                                   #sixteen times to generate 16 keys for each round
            lPostShiftArr = self.left_shift(leftKeyArray, i)
            rPostShiftArr = self.left_shift(rightKeyArray, i)   #calling the left_shift function
            keyValueArrPostShift = []
            keyValueArrPostShift.append(lPostShiftArr)
            keyValueArrPostShift.append(rPostShiftArr)
            keyValueArrPostShift = [ x for y in keyValueArrPostShift for x in y]    
            pc2KeyArr = self.permute(keyValueArrPostShift, PC_2)    
            self.keylist.append(pc2KeyArr)
            leftKeyArray = lPostShiftArr
            rightKeyArray = rPostShiftArr
            i += 1
        return self.keylist
        
    def XOR(self, a, b):
        # xor function - This function is complete
        return [i^j for i,j in zip(a,b)]

    def performRound(self,left,right,key):
        # Performs a single round of the DES algorithm
        ### YOUR CODE HERE ###
        
        #Expansion using E table
        expandedArr = []
        i = 0
        for i in range(0,48):
            expandedArr.append(right[E[i]-1])               #converting the 32bit right array to 48bit by using Expansion table
            i += 1
        afterExpXorValueArr = self.XOR(expandedArr,key)     #calling XOR function   

        subsValueArr = self.sbox_substition(afterExpXorValueArr)    #calling the substitution function 
        
        #permutation after substitution
        subsPermutedArr = self.permute(subsValueArr, P)
        nextRight = self.XOR(left, subsPermutedArr)
        nextLeft = right
        outputArr = []
        outputArr.append(nextLeft)
        outputArr.append(nextRight)
        return outputArr


    def performRounds(self, text, keys):
        # This function is used by the encrypt and decypt functions.
        # keys - A list of keys used in the rounds
        # text - The orginal text that is converted.
        ### YOUR CODE HERE ###
        leftArr = text[:32]
        rightArr = text[32:]                    #splitting the plaintext bits to two halves
        i = 0
        for i in range(0,16):                   #performing 16 rounds
            newTextArr = self.performRound(leftArr, rightArr, keys[i])
            leftArr = newTextArr[0]
            rightArr = newTextArr[1]
            i += 1
        finalLeft = rightArr
        finalRight = leftArr
        finalRoundArr = []
        finalRoundArr.append(finalLeft)
        finalRoundArr.append(finalRight)
        finalRoundArr = [ x for y in finalRoundArr for x in y]

        finalBitArray = self.permute(finalRoundArr, InvP)
        return finalBitArray
        

    def permute(self, bits, table):
        # Use table to permute the bits
        ### YOUR CODE HERE ###
        permutedArr = []
        i = 0
        for i in range(0,len(table)):
            permutedArr.append(bits[table[i]-1])
            i += 1
        return permutedArr

    def sbox_substition(self, bits):
        # Apply sbox subsitution on the bits
        ### YOUR CODE HERE ###
         #Substitution using SBOX
        beforeSubsArr = [bits[:6], bits[6:12], bits[12:18], bits[18:24], bits[24:30], bits[30:36], bits[36:42], bits[42:]]
        subsValueArr = []
        i = 0
        for i in range(0,8):
            block = i
            rowArr = []
            colArr = []
            rowArr.append(beforeSubsArr[i][0])      
            rowArr.append(beforeSubsArr[i][5])          #taking the zeroth bit and the fifth bit of each sixbit array for row
            colArr.append(beforeSubsArr[i][1])
            colArr.append(beforeSubsArr[i][2])
            colArr.append(beforeSubsArr[i][3])
            colArr.append(beforeSubsArr[i][4])          #taking the one,two,three and fourth bits of each sixbit array for column
            rowVal = ''.join(str(x) for x in rowArr)    #joining the bits  
            colVal = ''.join(str(x) for x in colArr)
            row = int(rowVal, 2)                        #converting the bits to integer number
            column = int(colVal, 2)
            subsValue = SBOX[block][row][column]        
            subsValueArr.append([int(a) for a in str('{0:04b}'.format(subsValue))]) #converts the number to 4 bit binary
            i += 1
        subsValueArr = [ x for y in subsValueArr for x in y]    #splits the each 4 bit binary to single bit elements in the array        
        return subsValueArr
        
        
    def encrypt(self):
        # Calls the performrounds function.
        ### YOUR CODE HERE ###
        plainBitsPermuted = self.permute(self.plaintext, IP)
        cypherArr = self.performRounds(plainBitsPermuted, self.keylist)
        cypherText = bitarray_to_str(cypherArr)
        return cypherText
        
    def decrypt(self, ciphertext):
        # Calls the performrounds function.
        ### YOUR CODE HERE ###
        cypherTextBit = str_to_bitarray(ciphertext)
        cypherBitsPermuted = self.permute(cypherTextBit, IP)
        reverseKeylist = list(reversed(self.keylist))       #sending the keys in the reverse order
        plainArr = self.performRounds(cypherBitsPermuted, reverseKeylist)
        plainText = bitarray_to_str(plainArr)
        return plainText

def pad_zeros(plaintext):
    #sending blocks of 8bytes each time to the DES class
    text =[]
    ciphertext = []
    print("Plaintext after padding is ",plaintext)
    print("Length of plaintext after padding zeros is ",len(plaintext))
    multiples = [plaintext[i: i+8] for i in range(0, len(plaintext), 8)]
    for i in range(0, len(multiples)):
        des = DES(multiples[i], key)
        ciphertext.append(des.encrypt())
        text.append(des.decrypt(ciphertext[i]))
        i += 1
    ciphertext = ''.join(ciphertext)
    text = [ x for y in text for x in y]
    output = []
    output.append(ciphertext)
    output.append(text)
    return output
        

if __name__ == '__main__':
    key = "9xc83vgs"
    plaintext= "Hi worldd"
    print("\nGiven Plaintext is ",plaintext)
    print("Length of given plaintext is ",len(plaintext))
    if (len(plaintext) % 8) == 0:
        des = DES(plaintext, key)
        ciphertext = des.encrypt() 
        text = des.decrypt(ciphertext) 
        text = ''.join(text)
        print("Encrypted ciphertext is ", ciphertext)
        print("Decrypted text is ", text)
    else:
        #padding zeros to the end of the string for non multiples of 8
        plaintext = plaintext.ljust((len(plaintext) + (8  - (len(plaintext) % 8))), '0')    
        plaintext = pad_zeros(plaintext)
        ciphertext = plaintext[0]
        text = plaintext[1]
        while '0' in text:
            text.remove('0')        #removing zeros from the plaintext after decryption
        text = ''.join(text)
        print("Ciphertext is ", ciphertext)
        print("Decryptedtext is ", text , "\n")

