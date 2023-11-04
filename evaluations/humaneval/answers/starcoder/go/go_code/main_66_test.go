package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestDigitSum(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(0, Digitsum(""))
    assert.Equal(131, Digitsum("abAB"))
    assert.Equal(67, Digitsum("abcCd"))
    assert.Equal(69, Digitsum("helloE"))
    assert.Equal(131, Digitsum("woArBld"))
    assert.Equal(153, Digitsum("aAaaaXa"))
    assert.Equal(151, Digitsum(" How are yOu?"))
    assert.Equal(327, Digitsum("You arE Very Smart"))
}
